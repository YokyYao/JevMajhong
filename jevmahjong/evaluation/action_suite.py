"""Small held-wall action panel; pair labels are sampled outcomes, not an oracle."""
from collections import defaultdict
import torch
from jevmahjong.training.sampling import OPERATIONS, stratum


@torch.no_grad()
def evaluate(heads, rows, per_direction=8):
    buckets = defaultdict(list)
    for row in rows:
        category, direction = stratum(row)
        if category in OPERATIONS and direction in ('positive', 'negative'):
            buckets[category, direction].append(row)
    result = {}
    for category in OPERATIONS:
        result[category] = {}
        for direction in ('positive', 'negative'):
            selected = sorted(buckets[category, direction], key=lambda r: tuple(r['episode_key']))[:per_direction]
            correct = 0; operation_choices = 0
            for row in selected:
                head = heads[row['scope']]
                z = head(row['decide'], row['options'])
                margin = float(z[row['action']] - z[row['teacher_action']])
                correct += int(margin * row['fast_advantage'] > 0)
                selected_index = max(row['eligible'], key=lambda i: float(z[i]))
                kinds = [v['type'] for v in row['public_request']['questions']['action']['criteria'].values()]
                operation_choices += int(kinds[selected_index] == category)
            result[category][direction] = {'cases': len(selected), 'pair_order_correct': correct,
                'pair_order_accuracy': correct / len(selected) if selected else None,
                'operation_selected_from_full_legal_set': operation_choices,
                'requested_cases': per_direction, 'coverage_complete': len(selected) >= per_direction}
    return {'actions': result, 'role': 'held-wall validation diagnostic; not independent full-game strength',
        'missing_coverage_is_not_a_pass': True}
