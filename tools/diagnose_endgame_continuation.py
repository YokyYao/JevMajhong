"""Paired training-only audit of the continuation policy assumed by labels."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def select_cases(records):
    import hashlib
    rows=[r for r in records if r['_meta'].get('split')=='train'
          and r['_meta'].get('source_initial_placement')==4
          and r['questions']['action']['src']=='endgame_terminal_win']
    chosen=[];groups=set()
    for row in sorted(rows,key=lambda r:hashlib.sha256(
            ('continuation-scope-pilot-v1:'+r['_meta']['id']).encode()).hexdigest()):
        if row['_meta']['group_id'] not in groups:
            chosen.append(row);groups.add(row['_meta']['group_id'])
            if len(chosen)==8:break
    return chosen


def audit(config):
    from riichienv import GameRule, RiichiEnv
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.state import build_request,legal_candidates,signature
    from jevmahjong.training.futures import replay_prefix,sample_future
    from jevmahjong.endgame.collect import digest,write
    from jevmahjong.endgame.context import serving_request,placement_objective,TRAILING_INPUT_VERSION
    from jevmahjong.endgame.prepare import read_jsonl
    from jevmahjong.endgame.futures import bundle_identity,finish_match
    from jevmahjong.endgame.policy import ModelAgent
    from jevmahjong.endgame.reference import escape_supported_keys,greedy_win_target
    from jevmahjong.evaluation.agents import EfficiencyAgent,CallingOpponent
    from jevmahjong.policy.routing import load_model
    root=Path(__file__).resolve().parents[1]
    run=Path(config['training_run']);out=Path(config['out'])
    if config.get('schema')!='continuation-scope-pilot-v1' or config.get('cases')!=8 or config.get('fit_registered') is not False:
        raise ValueError('Only the preregistered eight-training-case diagnostic is enabled')
    init=json.loads((run/'initialization-private.json').read_text())
    if not init['exact_parent_adapter_and_head'] or config['candidate_step']!=56:
        raise ValueError('Unexpected continuation ancestry or fixed checkpoint')
    source_config=json.loads((run/'config-private.json').read_text())
    if config['checkpoint']!=source_config['checkpoint']:
        raise ValueError('Original baseline differs from recorded trial')
    prior_audit=json.loads((run/'fit-audit-summary-private.json').read_text())
    if not prior_audit['actual_checkpoint_order_dataset_and_hook_exposure_match']:
        raise ValueError('Prior fit exposure/order was not verified')
    checkpoint=run/'training-run/checkpoints/step-000056'
    baseline_identity=bundle_identity(config['checkpoint']);candidate_identity=bundle_identity(checkpoint)
    chosen=select_cases(read_jsonl(run/'train.jsonl'))
    if len(chosen)!=8 or len({r['_meta']['group_id'] for r in chosen})!=8:
        raise ValueError('Diagnostic requires eight distinct training source groups')
    frozen_files={}
    for record in chosen:
        rid=record['_meta']['id'].split(':',1)[1]
        label_path=run/'labels'/(rid+'.json')
        label=json.loads(label_path.read_text())
        if label['worlds_per_action'] not in (4,16) or label['baseline_identity_sha256']!=baseline_identity['identity_sha256']:
            raise ValueError('Original paired worlds or baseline differ')
        for path in (label_path,run/'sources'/f"{label['source']['seed']}.json"):
            frozen_files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    paths=sorted(set(root.glob('jevmahjong/**/*.py'))|{Path(__file__).resolve(),root/'tools/train_endgame.py',
        root/'desktop/jev_sidecar/unified_policy.py',root/'desktop/jev_sidecar/call_policy.py',root/'desktop/jev_sidecar/tiles.py'})
    protocol={'schema':config['schema'],'configuration':config,'baseline_identity':baseline_identity,
        'continuation_identity':candidate_identity,'input_version':TRAILING_INPUT_VERSION,
        'selected_training_ids':[r['_meta']['id'] for r in chosen],
        'original_paired_files_sha256':frozen_files,
        'code_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'same_original_wall_seeds_and_world_counts':True,'training_only':True,'optimizer_updates':0}
    protocol_file=out/'protocol-private.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text())!=protocol:
        raise ValueError('Existing diagnostic protocol differs')
    write(protocol_file,protocol)
    original=ModelAgent(load_model(config['checkpoint']),placement_aware=False)
    class CountingContinuation:
        def __init__(self,model):
            self.agent=ModelAgent(model,placement_aware=True,input_version=TRAILING_INPUT_VERSION)
            self.declines=0
        def act(self,obs):
            offered=any(json.loads(a.to_mjai())['type']=='hora' for a in legal_candidates(obs))
            action=self.agent.act(obs)
            self.declines+=int(offered and json.loads(action.to_mjai())['type']!='hora')
            return action
    candidate=CountingContinuation(load_model(checkpoint))
    ordinary,caller=EfficiencyAgent(),CallingOpponent()
    results=[]
    for case_index,record in enumerate(chosen):
        rid=record['_meta']['id'].split(':',1)[1]
        label=json.loads((run/'labels'/(rid+'.json')).read_text());row=label['source']
        source=json.loads((run/'sources'/f"{row['seed']}.json").read_text())
        path=out/'cases'/(rid+'.json')
        if path.exists():
            result=json.loads(path.read_text())
            if result['protocol_sha256']!=digest(protocol):raise ValueError('Cached diagnostic changed')
            results.append(result);continue
        env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(row['seed'],'spaced-v1'))
        observations=replay_prefix(env,source['mjai_log'],row['step'])
        request,actions,_=build_request(observations[row['seat']]);request['state']['hand_number']=int(observations[row['seat']].kyoku_index)+1
        if digest(serving_request(request))!=row['request_sha256']:
            raise ValueError('Original public request replay differs')
        signatures=[signature(a) for a in actions];new=[[] for _ in actions]
        for world in range(label['worlds_per_action']):
            world_seed=int(digest([rid,world,'terminal-wall-v1'])[:16],16)
            sampled=sample_future(env,world_seed,'unseen')
            observations={p:o for p,o in sampled.clone().get_observations().items() if o.legal_actions()}
            others={p:(caller if p==row['seed']%4 else ordinary).act(o) for p,o in observations.items() if p!=row['seat']}
            for index,sig in enumerate(signatures):
                def branch():
                    current=sampled.clone();obs=current.get_observations()[row['seat']]
                    action={signature(a):a for a in legal_candidates(obs)}[sig]
                    return current,{**others,row['seat']:action}
                if case_index==0 and world==0:
                    current,initial=branch()
                    check=finish_match(current,initial,row['seat'],row['seed'],original,source_config['terminal_step_cap'])
                    if check!=label['outcomes'][index][world]:
                        raise ValueError('Original first-case first-world arm did not reproduce stored outcome exactly')
                current,initial=branch();candidate.declines=0
                outcome=finish_match(current,initial,row['seat'],row['seed'],candidate,source_config['terminal_step_cap'])
                outcome['continuation_declined_win_events']=candidate.declines
                new[index].append(outcome)
                write(out/'status-private.json',{'status':'paired_training_continuation_audit',
                    'cases_complete':len(results),'cases_total':8,'world':world+1,
                    'worlds':label['worlds_per_action'],'action':index+1,'actions':len(actions),'optimizer_updates':0})
        criteria=record['questions']['action']['criteria'];keys=list(criteria)
        old_means={k:sum(placement_objective(x['ranks'][row['seat']]) for x in outcomes)/len(outcomes)
                   for k,outcomes in zip(keys,label['outcomes'])}
        new_means={k:sum(placement_objective(x['ranks'][row['seat']]) for x in outcomes)/len(outcomes)
                   for k,outcomes in zip(keys,new)}
        old_escape={k:sum(x['ranks'][row['seat']]<4 for x in outcomes)/len(outcomes)
                    for k,outcomes in zip(keys,label['outcomes'])}
        new_escape={k:sum(x['ranks'][row['seat']]<4 for x in outcomes)/len(outcomes)
                    for k,outcomes in zip(keys,new)}
        win=next(k for k in keys if criteria[k]['type']=='hora');prior={k:float(k==win) for k in keys}
        old_target=greedy_win_target(criteria,old_means,escape_supported_keys(prior,old_escape,keys))
        new_target=greedy_win_target(criteria,new_means,escape_supported_keys(prior,new_escape,keys))
        result={'id':record['_meta']['id'],'protocol_sha256':digest(protocol),'worlds':label['worlds_per_action'],
            'old_means':old_means,'new_means':new_means,'old_escape':old_escape,'new_escape':new_escape,
            'old_risk_constrained_target':old_target,'new_risk_constrained_target':new_target,
            'old_target_greedy':max(old_target,key=old_target.get),'new_target_greedy':max(new_target,key=new_target.get),
            'win_key':win,'outcomes_under_candidate_continuation':new}
        write(path,result);results.append(result)
    for path,expected in protocol['code_sha256'].items():
        if hashlib.sha256((root/path).read_bytes()).hexdigest()!=expected:raise ValueError('Diagnostic code changed')
    for path,expected in frozen_files.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('Original paired data changed')
    summaries={}
    for kind in ('win','continue'):
        deltas=[];escape_deltas=[];events=0;branches=0
        for record,result in zip(chosen,results):
            criteria=record['questions']['action']['criteria']
            for key in result['old_means']:
                if (criteria[key]['type']=='hora')!=(kind=='win'):continue
                deltas.append(result['new_means'][key]-result['old_means'][key])
                escape_deltas.append(result['new_escape'][key]-result['old_escape'][key])
                index=list(criteria).index(key);outcomes=result['outcomes_under_candidate_continuation'][index]
                events+=sum(x['continuation_declined_win_events'] for x in outcomes);branches+=len(outcomes)
        summaries[kind]={'menus':len(deltas),'paired_mean_utility_change':sum(deltas)/len(deltas),
            'paired_mean_escape_change':sum(escape_deltas)/len(escape_deltas),
            'rollouts':branches,'continuation_declined_win_events':events}
    summary={'status':'paired_training_continuation_audit_complete','training_cases':len(results),
        'risk_constrained_target_greedy_changes':sum(r['old_target_greedy']!=r['new_target_greedy'] for r in results),
        'by_initial_action':summaries,'original_arm_first_case_first_world_reproduced':True,
        'training_only':True,'validation_or_development_cases_used':False,
        'limits':['Eight training groups and finite4/16 paired worlds; not model strength.',
            'Changing continuation alters both input/action path and weights; not an isolated refusal-only effect.',
            'Do not add this diagnostic subset to training as extra independent data.'],
        'optimizer_updates':0,'deployed':False}
    write(out/'summary-private.json',summary);write(out/'status-private.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Diagnostic requires a scheduled GPU job, never login')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    config=json.loads(args.config.read_text());sys.path.insert(0,config['upstream_source'])
    print(json.dumps(audit(config),indent=2),flush=True)


if __name__=='__main__':main()
