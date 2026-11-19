"""Audit training-only paired win/continue labels inside a GPU allocation.

The intervals describe finite sampled futures conditional on each fixed source;
they are not model-strength confidence intervals or expert correctness claims.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import sys


def paired_interval(differences, comparisons):
    from scipy.stats import t
    if len(differences) < 2 or comparisons < 1 or any(not math.isfinite(x) for x in differences):
        raise ValueError("Finite paired futures and positive comparison count required")
    n = len(differences)
    mean = sum(differences) / n
    variance = sum((x - mean) ** 2 for x in differences) / (n - 1)
    radius = t.ppf(1 - .05 / (2 * comparisons), n - 1) * math.sqrt(variance / n)
    return {"mean": mean, "conditional_mc_t_interval95_bonferroni": [mean - radius, mean + radius]}


def audit(config, partial=False):
    from jevmahjong.endgame.collect import digest, write
    from jevmahjong.endgame.context import placements, placement_objective, terminal_utility
    from jevmahjong.endgame.coverage import public_stratum
    from jevmahjong.endgame.win_value import quote_win
    out = Path(config['out'])
    manifest = json.loads((out / 'selected-positions-private.json').read_text())
    receipt = json.loads((out / 'manifest-status-private.json').read_text())
    if digest(manifest) != receipt['positions_sha256'] or config['future_worlds'] != 16:
        raise ValueError('Manifest identity or registered world count differs')
    training = [r for r in manifest if r['split'] == 'train']
    rows, counts, public_cells = [], Counter(), Counter()
    for row in training:
        if (public_stratum(row['public_request']) != row['coverage_stratum']
                or not row['coverage_stratum']['target']
                or quote_win(row['public_request']) != row['public_request']['state']['endgame_context']['current_win_settlement']):
            raise ValueError('Training row no longer matches public selection conditions')
        rid = digest([row['seed'], row['step'], row['seat'], row['request_sha256']])
        path = out / 'labels' / (rid + '.json')
        if not path.exists():
            if partial:
                continue
            raise ValueError('Training label incomplete; do not treat partial data as complete')
        data = json.loads(path.read_text())
        if (data['status'] != 'complete_terminal_menu'
                or data['source_request_sha256'] != row['request_sha256']
                or not data['continued_until_native_match_done']
                or data['worlds_per_action'] != 16):
            raise ValueError('Label protocol mismatch')
        question = data['record']['questions']['action']
        keys = list(question['criteria'])
        if len(keys) != len(data['outcomes']) or len(keys) != data['actual_actions']:
            raise ValueError('Action menu or complete outcomes differ')
        win = [i for i, key in enumerate(keys) if question['criteria'][key]['type'] == 'hora']
        if len(win) != 1 or len(keys) < 2:
            raise ValueError('Expected one legal win and at least one alternative')
        utility, points = [], []
        for outcomes in data['outcomes']:
            if len(outcomes) != 16:
                raise ValueError('Incomplete paired futures')
            for outcome in outcomes:
                if (not outcome['ledger_passed'] or placements(outcome['scores']) != outcome['ranks']
                        or outcome['hand_history']['final_scores'] != outcome['scores']
                        or outcome['utility'] != terminal_utility(outcome['ranks'][row['seat']])):
                    raise ValueError('Terminal score/rank/history audit failed')
            utility.append([placement_objective(x['ranks'][row['seat']]) for x in outcomes])
            points.append([x['scores'][row['seat']] for x in outcomes])
        comparisons = {}
        for index, key in enumerate(keys):
            if index == win[0]:
                continue
            differences = [a - b for a, b in zip(utility[index], utility[win[0]])]
            comparison = paired_interval(differences, len(keys) - 1)
            comparison['mean_final_point_difference'] = sum(a - b for a, b in zip(points[index], points[win[0]])) / 16
            comparisons[key] = comparison
        best = max(comparisons, key=lambda key: comparisons[key]['mean'])
        gain = comparisons[best]['mean']
        counts['positive_best_finite_mean' if gain > 1e-12 else 'negative_best_finite_mean' if gain < -1e-12 else 'neutral_best_finite_mean'] += 1
        counts['positive_lower_conditional_mc_bound'] += int(any(x['conditional_mc_t_interval95_bonferroni'][0] > 0 for x in comparisons.values()))
        counts['legal_actions_audited'] += len(keys)
        counts['complete_terminal_rollouts_audited'] += 16 * len(keys)
        public_cells[row['coverage_stratum']['cell']] += 1
        rows.append({'id': rid, 'group_id': row['group_id'], 'public_cell': row['coverage_stratum']['cell'],
                     'best_continuation_key': best, 'comparisons': comparisons})
    validation_files_present = sum((out / 'labels' / (digest([r['seed'], r['step'], r['seat'], r['request_sha256']]) + '.json')).exists()
                                   for r in manifest if r['split'] == 'validation')
    summary = {'status': 'complete_train_label_audit' if len(rows) == len(training) else 'partial_train_label_audit',
        'training_complete': len(rows), 'training_total': len(training),
        'validation_files_present_only_no_outcomes_inspected': validation_files_present,
        'counts': dict(counts), 'public_cells': dict(public_cells),
        'train_outcomes_only': True, 'best_finite_mean_is_optimistic_action_selection': True,
        'interval_scope': 'diagnostic t interval of16 sampled current-hand futures, within-menu Bonferroni; fixed hidden hands/future-hand RNG; not model strength',
        'negative_and_neutral_examples_retained': True, 'optimizer_updates': 0}
    write(out / 'train-label-audit-private.json', dict(summary, rows=rows))
    write(out / 'train-label-audit-summary-private.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-', 'login')):
        raise RuntimeError('Label auditing must run in an allocated GPU job, never on login')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    result = audit(json.loads(args.config.read_text()), args.partial)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
