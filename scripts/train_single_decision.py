"""Run the bounded existing trainer with explicitly tagged counterfactual collection."""
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from head_rl import fingerprint, reference_advantages, write
from single_decision_rl import SingleDecisionPolicy
from scripts import train_head_rl as runner


def intervention_advantages(controls, games, rows, replicas, scale):
    diagnostic = reference_advantages(controls, games, rows, replicas, scale)
    lookup = {(g['seed'], g['seat'], g['replica']): g for g in games}
    for row in rows:
        game = lookup[tuple(row['episode_key'])]
        if game['trainable_decisions'] != 1 or row['intervention_state_digest'] != game['intervention_state_digest']:
            raise ValueError('Only the one verified intervention may receive an advantage')
        if abs(row['advantage'] * scale * game['reward_scale'] - game['terminal_point_delta']) > 1e-6:
            raise ValueError('Intervention advantage differs from the paired terminal-point delta')
        if game['reused_control_outcome'] and game['terminal_point_delta'] != 0:
            raise ValueError('An unchanged action cannot have a nonzero counterfactual return')
    diagnostic.update(method='one_decision_Q_teacher_surrogate',
                      sampled_decisions=len(rows), actual_branch_simulations=sum(not g['reused_control_outcome'] for g in games),
                      nonzero_point_deltas=sum(g['terminal_point_delta'] != 0 for g in games),
                      full_trajectory_policy_gradient=False)
    return diagnostic


def protocol_write(path, data):
    if Path(path).name == 'protocol.json':
        data = copy.deepcopy(data)
        data['code_sha256'].update({name: fingerprint(ROOT / name) for name in
                                   ['single_decision_rl.py', 'scripts/train_single_decision.py',
                                    'semantic.py', 'seed_control.py']})
        data['training_data_mode'] = 'one independently sampled decision per frozen-policy continuation'
        data['objective_scope'] = 'sampled one-step Q_teacher surrogate, not full-trajectory on-policy PPO'
    write(path, data)


if __name__ == '__main__':
    runner.HeadPolicy = SingleDecisionPolicy
    runner.reference_advantages = intervention_advantages
    runner.write = protocol_write
    runner.main()
