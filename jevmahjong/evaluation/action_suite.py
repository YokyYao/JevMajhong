"""Small held-wall action panel; pair labels are sampled outcomes, not an oracle."""
from collections import defaultdict
import torch
from jevmahjong.training.sampling import OPERATIONS, stratum


@torch.no_grad()
def evaluate(heads, rows, per_direction=8, context=None):
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
            reference_correct = 0; reference_choices = 0
            reference_mass = []; candidate_mass = []; margin_gains = []
            for row in selected:
                delta = row.get('training_advantage', row['fast_advantage'])
                if context is not None:
                    from jevmahjong.policy.kan_context import probabilities
                    p = probabilities(row['public_request'],row['reference'].tolist(),context)
                    z = torch.tensor(p,dtype=torch.float64).clamp_min(1e-300).log()
                else:
                    head = heads[row['scope']]
                    z = head(row['decide'], row['options'])
                margin = float(z[row['action']] - z[row['teacher_action']])
                correct += int(margin * delta > 0)
                selected_index = max(row['eligible'], key=lambda i: float(z[i]))
                kinds = [v['type'] for v in row['public_request']['questions']['action']['criteria'].values()]
                operation_choices += int(kinds[selected_index] == category)
                reference = row['reference'].to(z.device).clamp_min(1e-30)
                prior_margin = float(reference[row['action']].log() - reference[row['teacher_action']].log())
                reference_correct += int(prior_margin * delta > 0)
                reference_choices += int(kinds[row['teacher_action']] == category)
                indices = [i for i, kind in enumerate(kinds) if kind == category]
                reference_mass.append(float(reference[indices].sum()))
                candidate_mass.append(float(z.softmax(-1)[indices].sum()))
                margin_gains.append((margin - prior_margin) * (1 if delta > 0 else -1))
            result[category][direction] = {'cases': len(selected), 'pair_order_correct': correct,
                'pair_order_accuracy': correct / len(selected) if selected else None,
                'operation_selected_from_full_legal_set': operation_choices,
                'reference_pair_order_correct': reference_correct,
                'reference_operation_selected': reference_choices,
                'reference_operation_probability_mean': sum(reference_mass) / len(selected) if selected else None,
                'candidate_operation_probability_mean': sum(candidate_mass) / len(selected) if selected else None,
                'reward_signed_pair_margin_gain_mean': sum(margin_gains) / len(selected) if selected else None,
                'requested_cases': per_direction, 'coverage_complete': len(selected) >= per_direction}
    return {'actions': result, 'role': 'held-wall validation diagnostic; not independent full-game strength',
        'missing_coverage_is_not_a_pass': True}


def main():
    import argparse
    import json
    from pathlib import Path
    from jevmahjong.core.io import digest, write
    from jevmahjong.policy.routing import make_head
    from jevmahjong.training.provenance import verify
    from jevmahjong.data.native_pairs import load_sources, select_decisions
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    fit = json.loads((args.fit_run / 'result.json').read_text())
    verify(json.loads((args.fit_run / 'protocol.json').read_text()))
    checkpoint = Path(fit['selected']['checkpoint'])
    if digest(checkpoint) != fit['selected']['sha256']:
        raise ValueError('Selected candidate changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    heads = {}
    for scope, role in [('ordinary', 'discard'), ('calling', 'call')]:
        meta = torch.load(Path(fit['config'][role + '_checkpoint']) / 'head.pt',
            map_location='cpu', weights_only=True)
        heads[scope] = make_head(meta, 'cpu').eval().requires_grad_(False)
        heads[scope].load_state_dict(saved['heads'][scope])
    partitions, hashes, _ = load_sources(fit.get('data',fit['config']['data']), saved['model_files'], fit['config'].get('reward'))
    partitions = select_decisions(partitions, fit['config'].get('decision_filter', 'all'), fit['config'].get('train_scopes'))
    if hashes != fit['data_sha256']:
        raise ValueError('Source shard manifest differs')
    if len(partitions['train']) != fit['train_rows'] or len(partitions['held']) != fit['held_rows']:
        raise ValueError('Whole-wall split coverage differs')
    result = {'model_id': fit['model_id'], 'checkpoint_sha256': digest(checkpoint),
        'strength_improvement_proven': False, 'human_test_opened': False}
    for key, rows in partitions.items():
        result[key] = evaluate(heads, rows, context=saved.get('kan_context'))
        result[key]['label_target'] = fit['config'].get('reward', {'mode': 'fast_win'})
        result[key]['role'] = key + '-wall action diagnostic; not independent full-game strength'
    write(args.out, result)


if __name__ == '__main__':
    main()
