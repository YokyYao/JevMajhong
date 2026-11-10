"""Validation-only exact meld-option audit; never opens test or selects weights."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest, metrics


def action_metrics(scored):
    report = metrics(scored)
    exact, predicted, support = Counter(), Counter(), Counter()
    for row in scored:
        support[row['gold']] += 1; predicted[row['pred']] += 1
        if row['exact']:
            if row['gold'] != row['pred']: raise ValueError('Exact action must have the same type')
            exact[row['gold']] += 1
    for kind, value in report['by_target'].items():
        value.update(exact_tp=exact[kind], predicted=predicted[kind],
                     exact_precision=exact[kind] / max(1, predicted[kind]),
                     exact_recall=exact[kind] / max(1, support[kind]))
    report['class_metrics_note'] = 'precision/recall/F1 use action type; exact_* additionally require the correct option/consumed tiles'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    import torch
    from head_rl import write
    from scripts.refit_call_query import load_cache
    from scripts.train_call_repair import make_head, logits_batch
    run, source, data, reference = (ROOT / cfg[k] for k in ('out', 'cache_source', 'data', 'reference'))
    selection = json.loads((run / 'selected.json').read_text())
    candidate = ROOT / selection['checkpoint']
    if digest(candidate) != selection['checkpoint_sha256']: raise ValueError('Selected candidate changed')
    if digest(reference / 'head.pt') != cfg['reference_head_sha256']: raise ValueError('Reference changed')
    audit = json.loads((data / 'provenance.json').read_text())
    if digest(data / 'validation.jsonl') != audit['files']['validation']: raise ValueError('Validation changed')
    rows = load_cache(source, data, 'validation', cfg['views'])
    meta = torch.load(reference / 'head.pt', map_location='cpu', weights_only=True)
    checkpoints = {'previous': ROOT / cfg['previous_call_checkpoint'], 'candidate': candidate}
    if cfg.get('include_reference'): checkpoints = {'reference': reference / 'head.pt', **checkpoints}
    for name, expected in cfg.get('expected_checkpoint_sha256', {}).items():
        if name not in checkpoints or digest(checkpoints[name]) != expected: raise ValueError('Fixed diagnostic checkpoint changed')
    result = {'split': 'validation_only', 'decisions_per_view': audit['counts']['validation'],
              'source_matches': len(audit['matches']['validation']), 'test_opened': False,
              'used_for_selection': False, 'checkpoint_sha256': {k: digest(p) for k, p in checkpoints.items()},
              'validation_sha256': audit['files']['validation'],
              'validation_role': cfg.get('validation_role', 'development validation'),
              'selection_origin': cfg.get('selection_origin', cfg['out']),
              'note': 'Three views reuse the same cases. Expert imitation is not a gameplay value estimate.'}
    for name, checkpoint in checkpoints.items():
        head = make_head(meta, 'cpu').eval().requires_grad_(False)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if name != 'reference' and saved['reference_head_sha256'] != cfg['reference_head_sha256']: raise ValueError('Checkpoint source mismatch')
        head.load_state_dict(saved['head']); views = {}
        for view in cfg['views']:
            selected = [r for r in rows if r['view'] == view]; scored = []
            with torch.no_grad():
                for start in range(0, len(selected), 64):
                    batch = selected[start:start + 64]
                    probabilities = logits_batch(head, batch, 'cpu').softmax(-1)
                    for row, p in zip(batch, probabilities):
                        pred, gold = int(p.argmax()), row['label']
                        scored.append({'gold': row['types'][gold], 'pred': row['types'][pred],
                                       'exact': pred == gold, 'target_probability': float(p[gold])})
            views[view] = action_metrics(scored)
        result[name] = views
    write(run / 'exact-validation-audit.json', result)
    print(json.dumps({name: {kind: result[name]['native']['by_target'][kind]
                            for kind in ('chi', 'pon')} for name in checkpoints}), flush=True)


if __name__ == '__main__': main()
