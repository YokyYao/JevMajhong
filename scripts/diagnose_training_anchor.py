"""Post-hoc third arm: decompose a failed candidate versus its training anchor.

Reuses already exposed confirmation streams for diagnosis only. Never promotes,
selects another checkpoint, or treats this as a new independent final test.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from head_rl import HeadPolicy, fingerprint, write
from scripts.audit_score_confirmation import compare
from scripts.train_head_rl import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    source = json.loads((args.source/'protocol.json').read_text())
    training = json.loads((args.source/'source-training-protocol.json').read_text())
    selected = json.loads((args.source/'confirmation-selection.json').read_text())
    completed = json.loads((args.source/'confirmation-result.json').read_text())
    if completed['status'] != 'complete':
        raise ValueError('The fixed original comparison must finish before post-hoc diagnosis')
    anchor = source['source_training_reference']
    if anchor != training['config']['reference']:
        raise ValueError('Training anchor provenance differs')
    if fingerprint(ROOT/anchor/'head.pt') != training['reference_head_sha256']:
        raise ValueError('Training anchor weights changed')
    if fingerprint(ROOT/selected['checkpoint']/'head.pt') != selected['candidate_head_sha256']:
        raise ValueError('Original selected candidate weights changed')
    for name, expected in source['code_sha256'].items():
        if fingerprint(ROOT/name) != expected:
            raise ValueError(f'Original numerical code changed: {name}')
    config = dict(source['config'])
    config.update(reference=anchor, development_seed_start=config['confirmation_seed_start'],
                  development_seeds=config['confirmation_seeds'], out=str(args.out))
    provenance = {'purpose': 'posthoc_diagnosis_only', 'original_confirmation_already_exposed': True,
                  'can_qualify_or_promote': False, 'source': str(args.source), 'config': config,
                  'anchor_head_sha256': training['reference_head_sha256'],
                  'candidate_head_sha256': selected['candidate_head_sha256'],
                  'code_sha256': {**source['code_sha256'], 'scripts/diagnose_training_anchor.py': fingerprint(Path(__file__))}}
    if args.out.exists():
        if not args.resume or json.loads((args.out/'protocol.json').read_text()) != provenance:
            raise ValueError('Preserve a diagnostic run; resume only its unchanged protocol')
    else:
        args.out.mkdir(parents=True)
        write(args.out/'protocol.json', provenance)
    write(args.out/'status.json', {'status': 'evaluating_training_anchor', 'purpose': 'posthoc_diagnosis_only'})
    policy = HeadPolicy(config)
    _, anchor_rows = evaluate(policy, args.out/'anchor', config, time.monotonic()+100*60, frozen=True)
    rows = {name: [json.loads(s) for s in (args.source/f'confirmation-{name}'/'games.jsonl').read_text().splitlines()]
            for name in ['candidate', 'reference']}
    anchor_vs_original = compare(anchor_rows, rows['reference'], config)
    candidate_vs_anchor = compare(rows['candidate'], anchor_rows, config)
    original_comparison = compare(rows['candidate'], rows['reference'], config)
    if abs(anchor_vs_original['primary_delta'] + candidate_vs_anchor['primary_delta'] - original_comparison['primary_delta']) > 1e-8:
        raise ValueError('Three-arm point deltas do not decompose')
    result = {'status': 'complete', 'purpose': 'posthoc_diagnosis_only', 'can_qualify_or_promote': False,
              'original_confirmation_qualified': completed['qualified'], 'default_policy_changed': False,
              'anchor': anchor, 'candidate': selected['checkpoint'],
              'anchor_vs_original': {k: anchor_vs_original[k] for k in ['candidate','reference','primary_delta','primary_delta_ci95']},
              'candidate_vs_anchor': {k: candidate_vs_anchor[k] for k in ['candidate','reference','primary_delta','primary_delta_ci95']},
              'candidate_vs_original': {k: original_comparison[k] for k in ['primary_delta','primary_delta_ci95']},
              'note': 'Exploratory decomposition on exposed streams, not another final test; future training needs fresh evaluation streams.'}
    write(args.out/'diagnosis.json', result)
    write(args.out/'status.json', {'status': 'complete', 'purpose': 'posthoc_diagnosis_only'})
    print('training_anchor_diagnosis '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
