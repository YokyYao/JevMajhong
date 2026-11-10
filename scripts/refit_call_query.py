"""Importance-weighted, low-LR continuation using audited frozen feature caches."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from call_policy import digest, is_call_request, request_hash
from head_rl import write
from scripts.train_call_repair import evaluate, logits_batch, make_head


def class_weights(priors, quotas):
    if set(priors) != {'chi', 'pon', 'none'} or abs(sum(priors.values()) - 1) > 1e-8:
        raise ValueError('Invalid natural class priors')
    if any(v <= 0 for v in [*priors.values(), *quotas.values()]): raise ValueError('Missing prior class')
    total = sum(quotas.values())
    return {k: priors[k] / (quotas[k] / total) for k in priors}


def load_cache(source, data, split, views):
    records = [json.loads(s) for s in (data / f'{split}.jsonl').read_text().splitlines()]
    expected = {(r['_meta']['id'], view): r for r in records for view in views}
    rows = []
    for p in sorted((source / 'features').glob(f'{split}-*.pt')):
        rows.extend(torch.load(p, map_location='cpu', weights_only=True))
    if len(rows) != len(expected) or len({(r['id'], r['view']) for r in rows}) != len(rows):
        raise ValueError('Incomplete or duplicated cache')
    for row in rows:
        rec = expected.get((row['id'], row['view']))
        if rec is None or not is_call_request(rec) or request_hash(rec) != rec['_meta']['text_sha256']:
            raise ValueError('Cache identity mismatch')
        q = rec['questions']['action']; keys = list(q['criteria'])
        if row['group'] != rec['_meta']['group_id'] or row['label'] != keys.index(q['label']) or row['types'] != [v['type'] for v in q['criteria'].values()]:
            raise ValueError('Cached feature/label alignment mismatch')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    source, data, out = (ROOT / cfg[k] for k in ('cache_source', 'data', 'out'))
    protocol = json.loads((source / 'protocol.json').read_text())
    audit = json.loads((data / 'provenance.json').read_text())
    if digest(data / 'provenance.json') != protocol['data_provenance_sha256']: raise ValueError('Source data provenance changed')
    for split in ('train', 'validation'):
        if digest(data / f'{split}.jsonl') != audit['files'][split]: raise ValueError('Source data changed')
    # Explicitly do not load the old opened test cache/labels.
    old_runtime_audit = json.loads((source / 'runtime-audit.json').read_text())
    if old_runtime_audit['native_feature_probability_checks'] != 15360 or not old_runtime_audit['reference_files_unchanged']:
        raise ValueError('Source caches lack native probability audit')
    reference = ROOT / cfg['reference']
    if digest(reference / 'head.pt') != cfg['reference_head_sha256']: raise ValueError('Reference changed')
    for filename, h in protocol['reference_files'].items():
        if digest(reference / filename) != h: raise ValueError('Frozen backbone inputs changed')
    prior = json.loads((ROOT / cfg['prior_data'] / 'provenance.json').read_text())
    if not prior['complete'] or prior['validation_or_test_matches_opened'] != 0: raise ValueError('Prior audit incomplete')
    old_groups = set(m for split in audit['matches'].values() for m in split)
    if old_groups & set(prior['matches']['train']): raise ValueError('Prior estimation reuses prior experiment matches')
    weights = class_weights(prior['priors'], cfg['train_quotas'])
    train = load_cache(source, data, 'train', cfg['views']); validation = load_cache(source, data, 'validation', cfg['views'])
    meta = torch.load(reference / 'head.pt', map_location='cpu', weights_only=True)
    warm = ROOT / cfg['warm_start']; warm_meta = torch.load(warm, map_location='cpu', weights_only=True)
    if warm_meta['reference_head_sha256'] != cfg['reference_head_sha256']: raise ValueError('Warm-start reference mismatch')
    out.mkdir(parents=True, exist_ok=False)
    p = {'config': cfg, 'importance_weights': weights, 'natural_priors': prior['priors'],
        'prior_provenance_sha256': digest(ROOT / cfg['prior_data'] / 'provenance.json'),
        'cache_protocol_sha256': digest(source / 'protocol.json'), 'warm_start_sha256': digest(warm),
        'cache_sha256': {p.name: digest(p) for split in ('train', 'validation') for p in (source / 'features').glob(f'{split}-*.pt')},
        'code_sha256': {n: digest(ROOT / n) for n in ['scripts/refit_call_query.py', 'scripts/train_call_repair.py', 'call_policy.py']},
        'test_opened': False, 'test_policy': 'old test is already opened and not reused as independent evidence'}
    write(out / 'protocol.json', p)
    torch.manual_seed(cfg['seed']); rng = random.Random(cfg['seed'])
    head = make_head(meta, 'cpu'); head.load_state_dict(warm_meta['head']); head.requires_grad_(False); head.q.requires_grad_(True)
    warm_report = evaluate(head, validation, 'cpu'); write(out / 'warm-start-validation.json', warm_report)
    baseline = json.loads((source / 'validation-baseline.json').read_text())
    optimizer = torch.optim.AdamW(head.q.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    best_report, best_step, stale = warm_report, 0, 0
    best_state = copy.deepcopy(head.state_dict()); curve = [{'step': 0, **warm_report}]
    for step in range(1, cfg['max_steps'] + 1):
        batch = rng.sample(train, min(cfg['batch_size'], len(train)))
        labels = torch.tensor([r['label'] for r in batch])
        importance = torch.tensor([weights[r['types'][r['label']]] for r in batch])
        losses = torch.nn.functional.cross_entropy(logits_batch(head, batch, 'cpu'), labels, reduction='none')
        loss = (losses * importance).mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.q.parameters(), 1.); optimizer.step()
        if step % cfg['eval_every']: continue
        report = evaluate(head, validation, 'cpu')
        curve.append({'step': step, 'train_loss': float(loss.detach()), **report})
        state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if any(not torch.equal(state[k], meta['head'][k]) for k in ('k.weight', 'k.bias')): raise ValueError('Frozen key changed')
        checkpoint = out / 'checkpoints' / f'step-{step:06d}'; checkpoint.mkdir(parents=True)
        torch.save({'head': state, 'reference_head_sha256': cfg['reference_head_sha256'],
            'scope': ['chi', 'pon', 'none'], 'step': step, 'method': 'natural-prior weighted query continuation'}, checkpoint / 'call-head.pt')
        native = report['views']['native']['by_target']
        eligible = native['chi']['recall'] > 0 and native['pon']['recall'] > 0
        better = eligible and (report['nll'], -report['macro_f1'], step) < (best_report['nll'], -best_report['macro_f1'], best_step)
        if better: best_report, best_step, best_state, stale = report, step, state, 0
        else: stale += 1
        write(out / 'validation-curve.json', curve)
        status = {'status': 'refitting_natural_prior_query', 'step': step, 'best_step': best_step,
            'validation_nll': report['nll'], 'validation_macro_f1': report['macro_f1'], 'stale_checks': stale}
        write(out / 'status.json', status); print(json.dumps(status), flush=True)
        if stale >= cfg['patience']: break
    selected_path = out / 'selected-call-head.pt'
    torch.save({'head': best_state, 'reference_head_sha256': cfg['reference_head_sha256'],
        'scope': ['chi', 'pon', 'none'], 'step': best_step, 'method': 'natural-prior weighted query continuation'}, selected_path)
    selected = {'step': best_step, 'checkpoint': str(selected_path.relative_to(ROOT)),
        'checkpoint_sha256': digest(selected_path), 'validation': best_report, 'baseline': baseline,
        'warm_start_validation': warm_report, 'test_opened_at_selection': False,
        'selection': 'minimum natural validation NLL with nonzero native chi/pon recall, macroF1 tie break'}
    write(out / 'selected.json', selected)
    write(out / 'training-result.json', {'status': 'refit_complete' if best_step else 'no_refit_improvement',
        'selected': selected, 'deployed': False, 'test_opened': False, 'importance_weights': weights})
    write(out / 'status.json', {'status': 'refit_complete' if best_step else 'no_refit_improvement',
        'best_step': best_step, 'deployed': False, 'test_opened': False})


if __name__ == '__main__': main()
