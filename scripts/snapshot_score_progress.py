"""Read-only score progress: report only complete comparisons, never pick a winner."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clean_policy_evaluation import aggregate, SUITES
from fresh_score_curve import policy_name


def complete_rows(path):
    """A concurrently appended last line is not a complete game record."""
    blob = path.read_bytes(); end = blob.rfind(b'\n') + 1
    prefix = blob[:end]
    return [json.loads(line) for line in prefix.splitlines() if line], len(blob) - end, hashlib.sha256(prefix).hexdigest()


def snapshot(rows, cfg, protocol, kind):
    if kind == 'curve':
        steps = ([protocol['selected_confirmation_step']] if cfg['stage'] == 'independent_confirmation'
            else sorted(int(s) for s in protocol['checkpoints']))
        parent, names = 'parent', ['parent', *(policy_name(s) for s in steps)]
    else:
        parent, names = 'reference', ['reference', 'candidate']
    expected = {(suite, seed, seat, name) for suite in SUITES
        for seed in range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds'])
        for seat in range(4) for name in names}
    actual = {(r['suite'], r['seed'], r['seat'], r['policy']) for r in rows}
    if len(actual) != len(rows) or not actual <= expected:
        raise ValueError('Duplicate or foreign progress record')
    for row in rows:
        if (not math.isfinite(row['score']) or row['rank'] not in (1, 2, 3, 4)
                or row['hands'] < 1 or not row['settlement_ledger_passed']
                or sum(row['point_components'].values()) != row['score'] - 25000
                or not 0 <= row['hand_wins'] <= row['hands'] or not 0 <= row['dealins'] <= row['hands']):
            raise ValueError('Progress contains invalid full-game settlement/counts')
    counts = Counter(r['policy'] for r in rows)
    wall_counts = Counter((r['policy'], r['seed']) for r in rows)
    per_wall = len(SUITES) * 4; per_policy = cfg['seeds'] * per_wall
    paired_walls = {}; completed = {}
    for name in names:
        if name == parent: continue
        paired_walls[name] = sum(wall_counts[parent, seed] == per_wall and wall_counts[name, seed] == per_wall
            for seed in range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds']))
        # Do not turn a few early walls into a declared improvement. Only the
        # complete registered cohort receives a numeric comparison here.
        if counts[parent] == per_policy and counts[name] == per_policy:
            paired = [dict(r, policy='reference' if r['policy'] == parent else 'candidate')
                for r in rows if r['policy'] in (parent, name)]
            completed[name] = aggregate(paired, cfg)
    return {'records': len(rows), 'expected_records': len(expected),
        'games_per_policy': {name: counts[name] for name in names},
        'expected_games_per_policy': per_policy, 'complete_paired_walls': paired_walls,
        'registered_wall_streams': cfg['seeds'], 'complete_comparisons': completed,
        'all_registered_games_present': actual == expected,
        'final_selection_or_confirmation_authorized': False,
        'formal_checkpoint_source_and_final_audit_not_replaced': True,
        'liveness': 'not inferred from files; check the Slurm job separately'}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--kind', choices=('curve', 'paired'), required=True); args = p.parse_args()
    cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['config'] != cfg: raise ValueError('Progress config differs from registered evaluation')
    rows, tail, sha = complete_rows(out / 'games-private.jsonl')
    result = snapshot(rows, cfg, protocol, args.kind)
    result.update(observed_unix=time.time(), complete_record_prefix_sha256=sha,
        ignored_unterminated_tail_bytes=tail, human_test_opened=False)
    target = out / 'progress-snapshot-private.json'; tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2) + '\n'); tmp.replace(target)
    print(json.dumps({k: v for k, v in result.items() if k != 'complete_comparisons'}), flush=True)


if __name__ == '__main__': main()
