"""Measure what the completed head update learned, without further optimization."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from jevany.model import PointerHead
from head_rl import mixed_probabilities


def load_head(path):
    meta = torch.load(path, map_location='cpu', weights_only=False)
    if meta['head_residual_dim'] != 0 or meta['temperature'] != 1.:
        raise ValueError('Diagnostic supports the frozen plain pointer head at temperature1')
    dp, d = meta['head']['q.weight'].shape
    head = PointerHead(d, dp).cpu().eval().requires_grad_(False)
    head.load_state_dict(meta['head'])
    return head


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Preserve previous diagnostics')
    protocol = json.loads((args.source/'protocol.json').read_text())
    config = protocol['config']
    selected = json.loads((args.source/'confirmation-selection.json').read_text())
    checkpoint = ROOT/selected['checkpoint']
    start = args.source/'checkpoints/update-000000'
    rows = torch.load(args.source/'rollout-stage-01.pt', map_location='cpu', weights_only=False)
    initial_state = torch.load(start/'rl_state.pt', map_location='cpu', weights_only=False)
    final_state = torch.load(checkpoint/'rl_state.pt', map_location='cpu', weights_only=False)
    exposures = final_state['update'] * config['minibatch']
    if exposures > len(rows):
        raise ValueError('Diagnostic assumes the first single partial data pass')
    rng, order = random.Random(), list(range(len(rows)))
    rng.setstate(initial_state['python_rng']); rng.shuffle(order)
    if rng.getstate() != final_state['python_rng']:
        raise ValueError('Cannot reproduce the recorded first shuffle')
    used = set(order[:exposures])
    old, new = load_head(start/'head.pt'), load_head(checkpoint/'head.pt')
    grouped, surrogate = {}, {'used': [0., 0., 0], 'unused': [0., 0., 0]}
    with torch.inference_mode():
        for index, row in enumerate(rows):
            logits_old, logits_new = [h(row['decide'], row['options']) for h in [old, new]]
            p, q = [mixed_probabilities(logits, row['eligible'], config['exploration_uniform'])
                    for logits in [logits_old, logits_new]]
            if not torch.allclose(p, row['reference'], atol=2e-5, rtol=2e-4):
                raise ValueError('Stored behavior probability differs')
            a, teacher = row['action'], row['teacher_action']
            logp_delta = float(q[a].log() - p[a].log())
            label = 'positive' if row['advantage'] > 0 else 'negative' if row['advantage'] < 0 else 'zero'
            subset = 'used' if index in used else 'unused'
            key = subset + '_' + label
            margin_old = logits_old[row['eligible'][a]] - logits_old[row['eligible'][teacher]]
            margin_new = logits_new[row['eligible'][a]] - logits_new[row['eligible'][teacher]]
            grouped.setdefault(key, []).append({'logp_delta': logp_delta,
                'margin_delta': float(margin_new-margin_old), 'aligned': bool(logp_delta*row['advantage'] > 0)})
            surrogate[subset][0] += row['advantage']
            surrogate[subset][1] += float(torch.exp(q[a].log()-p[a].log())) * row['advantage']
            surrogate[subset][2] += 1
    aggregates = {key: {'states': len(values), 'mean_selected_action_logp_change': sum(v['logp_delta'] for v in values)/len(values),
                         'mean_sampled_minus_teacher_logit_margin_change': sum(v['margin_delta'] for v in values)/len(values),
                         'moves_in_reward_direction': sum(v['aligned'] for v in values)} for key, values in grouped.items()}
    record = {'training_only_diagnostic': True, 'new_updates': 0, 'actual_source_updates': final_state['update'],
              'used_states': len(used), 'unused_states': len(rows)-len(used), 'first_shuffle_exactly_reproduced': True,
              'groups': aggregates,
              'sampled_surrogate': {key: {'old_mean': x/max(1,n), 'new_mean': y/max(1,n),
                                         'change': (y-x)/max(1,n)} for key,(x,y,n) in surrogate.items()},
              'candidate_head_sha256': hashlib.sha256((checkpoint/'head.pt').read_bytes()).hexdigest(),
              'note': 'Unused states are from the SAME training wall streams, not an independent validation set. '
                      'Likelihood/surrogate movement is not full-match score improvement.'}
    args.out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
