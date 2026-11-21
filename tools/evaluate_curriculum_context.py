"""Complete East matches for a saved checkpoint; fixed public development grid."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def play(agent,seed,seat,profile):
    from riichienv import RiichiEnv,GameRule
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.agents import EfficiencyAgent,CallingOpponent
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevmahjong.training.rewards import fast_game_stats
    env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(seed,'spaced-v1'))
    obs=env.reset();ordinary=EfficiencyAgent();caller=CallingOpponent()
    for _ in range(30000):
        if env.done():break
        obs=env.step({p:(agent if p==seat else caller if profile=='calling' and p==(seat+1)%4 else ordinary).act(o)
                      for p,o in obs.items()})
    if not env.done():raise RuntimeError('Truncated East match')
    log=env.mjai_log;ledger=settlement_ledger(log,env.scores());choices=Counter(e['type'] for e in log if e.get('actor')==seat)
    return dict(seed=seed,seat=seat,profile=profile,rank=env.ranks()[seat],net_points=env.scores()[seat]-25000,
        final_scores=list(env.scores()),hands=ledger['hands'],wins=choices['hora'],
        dealin_hands=hand_dealin_count(log,seat),completed=True,ledger_passed=True,
        choices={k:choices[k] for k in ('chi','pon','ankan','daiminkan','kakan','reach')},
        **fast_game_stats(log,seat,.98),mjai_log=log)

def evaluate(registered,checkpoint,step,groups,baseline=False):
    if not os.environ.get('SLURM_JOB_ID') or not (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS')):raise RuntimeError('Scheduled GPU required')
    manifest=read(registered);c=manifest['curriculum'];sys.path.insert(0,manifest['source'])
    sys.path.insert(0,manifest['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.evaluation.agents import PolicyAgent
    from jevmahjong.training.metric_journal import MetricJournal,aggregate_complete_matches,export_csv
    if package_digest(Path(manifest['source'])/'jevmahjong')!=manifest['source_sha256']:raise ValueError('Frozen source changed')
    for path,h in manifest['inputs'].items():
        if sha(path)!=h:raise ValueError('Registered input changed: '+path)
    identity=bundle(checkpoint)
    if baseline and identity['identity_sha256']!=c['reference_identity']:raise ValueError('Wrong v8 reference')
    from tools.curriculum_input_contract import load_context_client,verify_checkpoint
    contract=None if baseline else verify_checkpoint(checkpoint,manifest,bundle)
    out=ROOT/c['out'];metrics=out/'metrics';metrics.mkdir(parents=True,exist_ok=True)
    label='reference-v8' if baseline else f'step-{step:06d}'
    directory=out/'evaluations'/label;directory.mkdir(parents=True,exist_ok=False)
    seeds=list(range(c['curve_seed_start'],c['curve_seed_start']+groups))
    write(directory/'protocol.json',dict(checkpoint=identity,seeds=seeds,profiles=c['opponent_profiles'],
        seats=4,groups=groups,kind='reused_development_not_confirmation',new_optimizer_updates=0,
        input_contract=contract,reference_input_unchanged=baseline))
    client=UnifiedPolicyClient(load_model(checkpoint))
    if not baseline:client=load_context_client(client,checkpoint,manifest,bundle)
    agent=PolicyAgent(client);rows=[];started=time.monotonic()
    with (directory/'matches-private.jsonl').open('x') as f:
        for seed in seeds:
            for profile in c['opponent_profiles']:
                for seat in range(4):
                    row=play(agent,seed,seat,profile);rows.append(row)
                    f.write(json.dumps(row,separators=(',',':'))+'\n');f.flush()
                    write(directory/'status.json',dict(status='running',complete=len(rows),total=groups*8,
                        seconds=time.monotonic()-started))
    summary=aggregate_complete_matches(rows)
    summary.update(checkpoint_sha256=identity['identity_sha256'],stage='reference-v8' if baseline else c['stage'],
        input_contract=contract,
        source='complete_saved_checkpoint_east_matches',grid='stage_end64' if groups==8 else 'curve16',
        ci95=None,ci_note='Development seed clusters are reported; no independent strength claim',
        discounted_own_wins_per_hand=sum(r['fast_reward_sum']/r['hands'] for r in rows)/len(rows),
        seven_pairs_counts_pending_separate_event_audit=True)
    if groups==8:
        # The first two seeds provide the periodic grid, without playing them twice.
        small=aggregate_complete_matches([r for r in rows if r['seed']<c['curve_seed_start']+c['curve_seed_groups']])
        summary['curve_grid_subset']=small
    if bundle(checkpoint)!=identity:raise ValueError('Evaluation changed checkpoint')
    write(directory/'summary.json',summary);write(directory/'status.json',dict(status='complete',games=len(rows)))
    if not baseline:
        journal=MetricJournal(metrics,c['run_id'],c['stage'],c['stage_offset'])
        fields={k:v for k,v in summary.items() if k!='stage'};journal.append('evaluation',step,**fields)
        export_csv(metrics)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registered',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--step',type=int,required=True);p.add_argument('--groups',type=int,choices=(2,8),required=True)
    p.add_argument('--baseline',action='store_true');a=p.parse_args();evaluate(a.registered,a.checkpoint,a.step,a.groups,a.baseline)
