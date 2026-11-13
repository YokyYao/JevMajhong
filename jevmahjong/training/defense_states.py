"""Capture every public open-tenpai defense opportunity from fresh controls."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import torch
from jevmahjong.core.io import digest,request_hash,write
from jevmahjong.core.state import build_request
from jevmahjong.data.native_states import METHOD,KAN_METHOD,validate
from jevmahjong.policy.defense import DefensePolicyAgent,is_defense_request,eligible_indices,request_eligible
from .collector import NativeCollector,action_key
from .provenance import snapshot,verify


class DefenseStateCollector(NativeCollector):
    def __init__(self,config,device='cuda'):
        if config.get('behavior_heads'):raise ValueError('Original frozen Play encoders and heads required')
        super().__init__(config,device);self.agent=DefensePolicyAgent(self.logged)

    def act(self,obs,frozen=False,stochastic=False,trace=None):
        if not frozen or stochastic:raise ValueError('Frozen greedy collection required')
        request=build_request(obs)[0] if trace is not None else None
        target=request is not None and is_defense_request(request)
        self.fast_inputs.clear();self.logged.request=self.logged.response=None;self.capture_active=target
        try:chosen=self.agent.act(obs)
        finally:self.capture_active=False
        if target:
            if len(self.fast_inputs)!=1 or self.fast_inputs[0][0]!='ordinary':raise ValueError('One ordinary feature capture required')
            from jevany.api import validate_distribution
            decide,options=self.fast_inputs[0][1]
            with torch.no_grad():q=self.policy_heads['ordinary'](decide,options).softmax(-1).detach().cpu()
            captured=self.logged.request
            if request_hash(captured)!=request_hash(request):raise ValueError('Captured public request changed')
            native,_=validate_distribution(self.logged.response['answers']['action']['probabilities'],list(captured['questions']['action']['criteria']))
            if not torch.allclose(q,torch.tensor(native),atol=2e-5,rtol=2e-4):raise ValueError('Native policy and cached head differ')
            _,actions,features=build_request(obs);eligible=eligible_indices(obs,actions,features)
            if eligible!=request_eligible(captured):raise ValueError('Public and native defense admission differ')
            teacher=next(i for i,a in enumerate(actions) if action_key(a)==action_key(chosen))
            if teacher!=max(eligible,key=lambda i:float(q[i])):raise ValueError('Corrected-Play teacher differs from actual action')
            trace.append({'scope':'ordinary','decide':decide.cpu(),'options':options.cpu(),'reference':q,
                'eligible':eligible,'teacher_action':teacher,'action_keys':[action_key(a) for a in actions],
                'public_request':copy.deepcopy(captured),'request_sha256':request_hash(captured)})
            self.checks+=1
        self.fast_inputs.clear();return chosen


def collect_native_inputs(cfg,resume=False,max_walls=None,*,collector=DefenseStateCollector,method=METHOD,scope='ordinary'):
    """One writer for audited, unlabeled native inputs from either policy scope."""
    out=Path(cfg['out'])
    total=cfg['training_seeds_per_stage']
    if max_walls is not None and not 1<=max_walls<=total:raise ValueError('Bounded wall prefix required')
    expected_filter='kan_windows' if method==KAN_METHOD else 'all' if scope=='calling' else 'open_tenpai_riichi'
    if cfg['train_scopes']!=[scope] or cfg['decision_filter']!=expected_filter:
        raise ValueError('Explicit native-state capture scope required')
    out.mkdir(parents=True,exist_ok=resume);lock=(out/'collection.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=snapshot(cfg);protocol['collection_method']=method
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:
        raise ValueError('Resume requires unchanged registered source, models and configuration')
    write(out/'protocol.json',protocol);shards=out/'walls-private';shards.mkdir(exist_ok=True)
    policy=collector(cfg);counts={'train':0,'held':0};rows_count=0
    collecting=('collecting_native_kan_states' if method==KAN_METHOD else
        'collecting_native_call_states' if scope=='calling' else 'collecting_native_defense_states')
    try:
        for completed,seed in enumerate(range(cfg['training_seed_start'],cfg['training_seed_start']+total),1):
            path=shards/f'{seed}.pt';receipt=path.with_suffix('.audit.json')
            if path.exists() or receipt.exists():
                if not resume or not path.exists() or not receipt.exists():raise ValueError('Inspect incomplete existing native state shard')
                if digest(path)!=json.loads(receipt.read_text())['sha256']:raise ValueError('Completed native shard changed')
                payload=torch.load(path,map_location='cpu',weights_only=False)
            else:
                controls=[];decisions=[];rows=[]
                for seat in range(4):
                    control,_=policy.game(seed,seat,frozen=True,replica=-1);controls.append(control)
                    saved=policy.controls.pop((seed,seat))
                    for replica,decision in enumerate(saved['decisions']):
                        row=copy.deepcopy(decision['row']);key=(seed,seat,replica)
                        row.update(episode_key=key,category=scope,
                            action=next(i for i in row['eligible'] if i!=row['teacher_action']),
                            source_state_digest=decision['digest'])
                        rows.append(row);decisions.append({'episode_key':key,'seed':seed,'seat':seat,'replica':replica,
                            'intervention_step':decision['step'],'intervention_state_digest':decision['digest'],
                            'hand_index':row['hand_index'],'draws_before':row['draws_before']})
                    write(out/'status.json',{'status':collecting,'walls_complete':completed-1,
                        'walls_total':total,'current_seat':seat,'states_from_complete_walls':rows_count})
                payload={'collection_method':method,'seed':seed,'controls':controls,'decisions':decisions,'rows':rows}
                validate(payload,seed,cfg);temporary=path.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(path)
                write(receipt,{'sha256':digest(path),'rows':len(rows),'controls':4})
            validate(payload,seed,cfg);rows_count+=len(payload['rows'])
            counts['held' if (seed-cfg['training_seed_start'])%4==3 else 'train']+=len(payload['rows'])
            state={'status':collecting,'walls_complete':completed,'walls_total':total,
                'states':rows_count,'state_splits':counts,'fresh_control_matches':completed*4,
                'counterfactual_rollouts':0,'optimizer_updates':0}
            write(out/'status.json',state)
            if max_walls is not None and completed>=max_walls and completed<total:
                verify(protocol);write(out/'status.json',{**state,'status':'partial_prefix_complete','resume_required':True});return
        verify(protocol)
        result={'status':'collection_complete','method':method,'walls':total,'rows':rows_count,
            'state_splits':counts,'fresh_control_matches':total*4,'counterfactual_rollouts':0,
            'optimizer_updates':0,'human_test_opened':False,'inputs_only_not_reward_labels':True}
        write(out/'collection-result.json',result);write(out/'status.json',result)
    finally:policy.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--max-walls',type=int)
    args=parser.parse_args();collect_native_inputs(json.loads(args.config.read_text()),args.resume,args.max_walls)


if __name__=='__main__':main()
