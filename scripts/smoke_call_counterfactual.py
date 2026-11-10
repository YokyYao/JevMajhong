"""One untouched training-wall GPU integration check; not a strength experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from call_counterfactual import CallInterventionPolicy
from call_policy import digest
from head_rl import write
from scripts.train_single_decision import intervention_advantages


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    reference = ROOT / cfg['reference']
    source = json.loads((ROOT / cfg['reference_training_protocol']).read_text())
    for name, h in source['reference_files'].items():
        if digest(reference / name) != h: raise ValueError('Frozen base changed')
    proposed = {cfg['training_seed_start']}
    if cfg['training_seeds_per_stage'] != 1: raise ValueError('This is a one-wall integration check')
    for path in (ROOT / 'runs').glob('*/protocol.json'):
        old = json.loads(path.read_text()).get('config', {})
        for start, length in [('training_seed_start', 'training_seeds_per_stage'),
                              ('development_seed_start', 'development_seeds'), ('confirmation_seed_start', 'confirmation_seeds')]:
            if start in old and length in old:
                count = old.get('max_stages', 1) if start == 'training_seed_start' else 1
                if proposed & set(range(old[start], old[start] + old[length] * count)):
                    raise ValueError(f'Implementation-check wall overlaps a declared range: {path}')
    out.mkdir(parents=True, exist_ok=False)
    code = ['call_counterfactual.py', 'single_decision_rl.py', 'head_rl.py', 'arena.py', 'semantic.py',
            'seed_control.py', 'call_policy.py', 'scripts/evaluate_call_repair.py', 'scripts/train_call_repair.py',
            'scripts/train_single_decision.py', 'scripts/smoke_call_counterfactual.py', 'vendor/JevAny/jevany/model.py']
    write(out / 'protocol.json', {'config': cfg, 'code_sha256': {name: digest(ROOT / name) for name in code},
        'code_tree_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip(),
        'reference_files': source['reference_files'], 'call_head_sha256': cfg['call_head_sha256'],
        'purpose': 'native integration smoke only; not statistical evidence or actor training',
        'actor_updates': 0, 'human_test_opened': False})
    write(out / 'status.json', {'status': 'loading_frozen_native_call_teacher', 'actor_updates': 0})
    policy = CallInterventionPolicy(cfg)
    games, rows = [], []
    try:
        seed, seat = cfg['training_seed_start'], cfg['seat']
        control, trace = policy.game(seed, seat, frozen=True, replica=-1)
        if trace or control['available_intervention_decisions'] < 1: raise ValueError('Chosen wall did not exercise a call')
        for replica in range(cfg['training_replicas']):
            game, trace = policy.game(seed, seat, training=True, replica=replica)
            if len(trace) != 1 or not set(trace[0]['types']) <= {'chi', 'pon', 'none'}:
                raise ValueError('Single call-only trace required')
            games.append(game); rows.extend(trace)
            state = {'status': 'checking_native_call_counterfactuals', 'completed_replicas': len(games),
                'total_replicas': cfg['training_replicas'], 'native_probability_checks': policy.checks, 'actor_updates': 0}
            write(out / 'status.json', state); print(json.dumps(state), flush=True)
        diagnostics = intervention_advantages([control], games, rows, cfg['training_replicas'], cfg['advantage_scale'])
        if policy.controls or policy.checks < 1 or diagnostics['actual_branch_simulations'] < 1:
            raise ValueError('Native replay/feature mechanism was not fully exercised')
        if digest(ROOT / cfg['call_checkpoint']) != cfg['call_head_sha256']: raise ValueError('Call teacher changed')
        if any(digest(reference / name) != h for name, h in source['reference_files'].items()): raise ValueError('Base changed during smoke')
        write(out / 'control-private.json', control); write(out / 'branches-private.json', games)
        torch.save(rows, out / 'call-traces-private.pt')
        result = {'status': 'native_call_counterfactual_smoke_passed', 'native_probability_checks': policy.checks,
            'replicas': len(games), 'branch_simulations': diagnostics['actual_branch_simulations'],
            'nonzero_point_deltas': sum(g['terminal_point_delta'] != 0 for g in games),
            'same_action_reuses': sum(g['reused_control_outcome'] for g in games),
            'single_legal_call_credit_verified': True, 'frozen_source_files_unchanged': True,
            'actor_updates': 0, 'human_test_opened': False, 'strength_improvement_proven': False,
            'training_labels_ready_for_fit': False, 'scope': 'One wall integration smoke; not a training/validation study'}
        write(out / 'result.json', result); write(out / 'status.json', result); print(json.dumps(result), flush=True)
    finally: policy.close()


if __name__ == '__main__': main()
