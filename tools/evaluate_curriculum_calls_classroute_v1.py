"""Complete matched East games using the frozen RI class selection method."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_riichi_class_policy_v1 import RiichiClassPolicyAgent, METHOD
from tools.evaluate_curriculum_riichi_class_route_v2 import status_write
from tools.evaluate_curriculum import play


def policy_contract(identity, reg):
    sealed = read(reg['curriculum']['parent_seal'])
    # config_sha256 is the inherited original RI-route experiment recipe,
    # not this child's trainer/checkpoint path. Runtime execution is the bound
    # selector implementation; only the actual parameter identity changes.
    spec = dict(sealed['policy'], parameter_identity=identity['identity_sha256'])
    if (spec['method'] != METHOD or spec['route_sha256'] != sha(ROOT/'tools/curriculum_riichi_class_policy_v1.py')
            or spec['runtime_source_sha256'] != reg['source_sha256']):
        raise ValueError('Explicit frozen RI execution route required')
    return spec, hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def evaluate(registered, checkpoint, step, groups):
    gpu_only(); reg = read(registered); c = reg['curriculum']
    if groups not in (2, 8) or step not in (0, 194, 388) or (groups == 8) != (step == 388):
        raise ValueError('Frozen matched16 process/final64 development schedule required')
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.training.metric_journal import MetricJournal, aggregate_complete_matches, export_csv
    if package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']: raise ValueError('Pinned runtime changed')
    for path, expected in reg['inputs'].items():
        if sha(path) != expected: raise ValueError('Registered input changed: '+path)
    identity = bundle(checkpoint); spec, policy_identity = policy_contract(identity, reg)
    if step == 0 and (identity['identity_sha256'] != c['parent_identity'] or policy_identity != c['parent_policy_identity']):
        raise ValueError('Step zero must be the exact distinct parent policy')
    directory = ROOT/c['out']/'evaluations'/f'step-{step:06d}'; directory.mkdir(parents=True, exist_ok=False)
    seeds = list(range(c['curve_seed_start'], c['curve_seed_start']+groups))
    write(directory/'protocol.json', dict(checkpoint=identity, policy=spec, policy_identity=policy_identity,
        seeds=seeds, profiles=c['opponent_profiles'], seats=4, groups=groups,
        kind='reused_development_not_confirmation', selector_code_sha256=spec['route_sha256']))
    agent = RiichiClassPolicyAgent(UnifiedPolicyClient(load_model(checkpoint)))
    games = []; started = time.monotonic()
    with (directory/'matches-private.jsonl').open('x') as stream:
        for seed in seeds:
            for opponent in c['opponent_profiles']:
                for seat in range(4):
                    row = play(agent, seed, seat, opponent); games.append(row)
                    stream.write(json.dumps(row, separators=(',', ':'))+'\n'); stream.flush()
                    status_write(directory/'status.json', dict(status='running', complete=len(games),
                        total=groups*8, seconds=time.monotonic()-started))
    summary = aggregate_complete_matches(games)
    summary.update(checkpoint_sha256=identity['identity_sha256'], policy_identity=policy_identity,
        selection_method=METHOD, stage=c['stage'], source='complete_saved_checkpoint_east_matches',
        grid='stage_end64' if groups == 8 else 'curve16', ci95=None,
        ci_note='Consumed development groups, not independent strength proof', diagnostics=dict(agent.diagnostics),
        discounted_own_wins_per_hand=sum(r['fast_reward_sum']/r['hands'] for r in games)/len(games))
    if groups == 8:
        summary['curve_grid_subset'] = aggregate_complete_matches([r for r in games if r['seed'] < c['curve_seed_start']+2])
    if bundle(checkpoint) != identity: raise ValueError('Evaluation changed parameters')
    write(directory/'summary.json', summary)
    status_write(directory/'status.json', dict(status='complete', games=len(games)))
    journal = MetricJournal(ROOT/c['out']/'metrics', c['run_id'], c['stage'], c['stage_offset'])
    journal.append('evaluation', step, **{k: v for k, v in summary.items() if k != 'stage'})
    export_csv(ROOT/c['out']/'metrics'); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--step', type=int, required=True)
    p.add_argument('--groups', type=int, required=True); a = p.parse_args()
    evaluate(a.registered.resolve(), a.checkpoint.resolve(), a.step, a.groups)
