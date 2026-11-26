"""Actual full-East saved-checkpoint development games for placement course."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_placement_runtime_v1 import require_registration, policy_contract, require_step0, evaluation_groups, matched_curve
from tools.curriculum_placement_full_win_policy_v1 import agent as placement_agent, METHOD
from tools.play_curriculum_placement_v1 import play
from tools.evaluate_curriculum_riichi_class_route_v2 import status_write


def evaluate(registered, checkpoint, step):
    gpu_only(); reg = read(registered); require_registration(reg); c = reg['curriculum']
    groups = evaluation_groups(reg, step)
    expected_checkpoint = ROOT/c['out']/'training-run/checkpoints'/f'step-{step:06d}'
    if checkpoint.resolve() != expected_checkpoint.resolve():
        raise ValueError('Only the actual course saved checkpoint at the registered step is eligible')
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.training.metric_journal import MetricJournal, aggregate_complete_matches, export_csv
    for path, expected in reg['inputs'].items():
        if sha(path) != expected: raise ValueError('Registered input changed: '+path)
    identity = bundle(checkpoint); spec, policy_id = policy_contract(identity, reg)
    if step == 0: require_step0(identity, reg, read(ROOT/c['out']/'warmstart-validation-private.json'))
    directory = ROOT/c['out']/'evaluations'/f'step-{step:06d}'
    directory.mkdir(parents=True, exist_ok=False)
    seeds = list(range(c['curve_seed_start'], c['curve_seed_start']+groups))
    write(directory/'protocol.json', dict(checkpoint=identity, policy=spec, policy_identity=policy_id,
        seeds=seeds, profiles=c['opponent_profiles'], seats=4, groups=groups,
        kind='consumed_development_not_confirmation', parent_cache_reused_as_step0=False))
    agent = placement_agent(UnifiedPolicyClient(load_model(checkpoint)))
    games = []; started = time.monotonic()
    with (directory/'matches-private.jsonl').open('x') as stream:
        for seed in seeds:
            for opponent in c['opponent_profiles']:
                for seat in range(4):
                    row = play(agent, seed, seat, opponent); games.append(row)
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                    status_write(directory/'status.json', dict(status='running', complete=len(games),
                        total=groups*8, seconds=time.monotonic()-started))
    summary, curve = matched_curve(games, reg, step, aggregate_complete_matches)
    common = dict(checkpoint_sha256=identity['identity_sha256'], policy_identity=policy_id, selection_method=METHOD,
        stage=c['stage'], source='complete_actual_saved_checkpoint_matches', ci95=None,
        ci_note='Consumed logical seed clusters, not independent strength proof', diagnostics=dict(agent.diagnostics))
    summary.update(common); curve.update(common)
    if bundle(checkpoint) != identity: raise ValueError('Evaluation modified model parameters')
    write(directory/'summary.json', summary); write(directory/'curve16-summary.json', curve)
    status_write(directory/'status.json', dict(status='complete', games=len(games)))
    journal = MetricJournal(ROOT/c['out']/'metrics', c['run_id'], c['stage'], c['stage_offset'])
    journal.append('evaluation', step, **{k: v for k, v in curve.items() if k != 'stage'})
    export_csv(ROOT/c['out']/'metrics'); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--step', type=int, required=True)
    a = p.parse_args(); evaluate(a.registered.resolve(), a.checkpoint.resolve(), a.step)
