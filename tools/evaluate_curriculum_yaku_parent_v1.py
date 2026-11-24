"""Complete saved yaku_course checkpoint matches on the frozen development grid."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_yaku_parent_runtime_v1 import require_registration, policy_contract, require_step0
from tools.curriculum_complete_legal_class_policy_v1 import CompleteLegalClassPolicyAgent, METHOD
from tools.evaluate_curriculum_riichi_class_route_v2 import status_write
from tools.evaluate_curriculum import play


def evaluate(registered, checkpoint, step):
    gpu_only(); reg = read(registered); require_registration(reg); c = reg['curriculum']
    if step not in (0, 625, 1250, 1875, 2500): raise ValueError('Only registered actual checkpoint points')
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.training.metric_journal import MetricJournal, aggregate_complete_matches, export_csv
    for p, h in reg['inputs'].items():
        if sha(p) != h: raise ValueError('Registered input changed: '+p)
    identity = bundle(checkpoint); spec, policy_id = policy_contract(identity, reg)
    if step == 0: require_step0(identity, reg, read(ROOT/c['out']/'warmstart-validation-private.json'))
    groups = 8 if step == 2500 else 2; directory = ROOT/c['out']/'evaluations'/f'step-{step:06d}'
    directory.mkdir(parents=True, exist_ok=False)
    seeds = list(range(c['curve_seed_start'], c['curve_seed_start']+groups))
    write(directory/'protocol.json', dict(checkpoint=identity, policy=spec, policy_identity=policy_id,
        seeds=seeds, profiles=c['opponent_profiles'], seats=4, groups=groups, kind='consumed_development_not_confirmation'))
    agent = CompleteLegalClassPolicyAgent(UnifiedPolicyClient(load_model(checkpoint))); games = []; started = time.monotonic()
    with (directory/'matches-private.jsonl').open('x') as stream:
        for seed in seeds:
            for opponent in c['opponent_profiles']:
                for seat in range(4):
                    row = play(agent, seed, seat, opponent); games.append(row)
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                    status_write(directory/'status.json', dict(status='running', complete=len(games), total=groups*8,
                        seconds=time.monotonic()-started))
    summary = aggregate_complete_matches(games)
    summary.update(checkpoint_sha256=identity['identity_sha256'], policy_identity=policy_id, selection_method=METHOD,
        stage=c['stage'], source='complete_actual_saved_checkpoint_matches', grid='stage_end64' if groups == 8 else 'curve16',
        ci95=None, ci_note='Consumed logical seed clusters, not independent strength proof', diagnostics=dict(agent.diagnostics),
        discounted_own_wins_per_hand=sum(r['fast_reward_sum']/r['hands'] for r in games)/len(games))
    if groups == 8: summary['curve_grid_subset'] = aggregate_complete_matches([r for r in games if r['seed'] < seeds[0]+2])
    if bundle(checkpoint) != identity: raise ValueError('Evaluation modified parameters')
    write(directory/'summary.json', summary); status_write(directory/'status.json', dict(status='complete', games=len(games)))
    journal = MetricJournal(ROOT/c['out']/'metrics', c['run_id'], c['stage'], c['stage_offset'])
    journal.append('evaluation', step, **{k: v for k, v in summary.items() if k != 'stage'})
    export_csv(ROOT/c['out']/'metrics'); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--step', type=int, required=True)
    a = p.parse_args(); evaluate(a.registered.resolve(), a.checkpoint.resolve(), a.step)
