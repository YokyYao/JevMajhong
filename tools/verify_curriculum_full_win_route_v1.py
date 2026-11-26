"""Actual complete-current-win-menu native/single-forward proof, not strength."""
from collections import Counter
import copy,json,os,socket,sys,tomllib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_placement_input_v1 import serving_request,training_record,SCHEMA
from tools.curriculum_placement_full_win_policy_v1 import CompleteWinPlacementPolicyAgent,METHOD
from tools.curriculum_placement_input_v1 import PlacementPolicyClient
from tools.verify_curriculum_rank_win_input_v1 import Selector

PUBLIC_FILES=('tools/curriculum_placement_full_win_policy_v1.py','tools/verify_curriculum_full_win_route_v1.py',
    'tests/core/test_curriculum_placement_full_win_policy_v1.py','cluster/curriculum-full-win-route-v1.slurm',
    'tools/play_curriculum_placement_v1.py')


def verify():
    if (not os.environ.get('SLURM_JOB_ID') or not (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
        or not socket.gethostname().startswith('ws-')):raise RuntimeError('Actual allocated GPU resources required')
    control_path=ROOT/'runs/curriculum-rank-win-controls-v1/registered-private.json';control=read(control_path)
    source=read(control['source_registration'])
    sys.path.insert(0,source['source']);sys.path.insert(0,source['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(1)
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.core.state import signature
    from jevany.api import SystemOneRequest,to_record
    from jevany.data import api_request
    from jevany.model import DecisionModel,load_preprocessor
    from tools.audit_curriculum_reused_terminal_v1 import restore
    from tools.curriculum_win_context import build_win_request
    previous_path=ROOT/'runs/curriculum-rank-win-input-v1/verification-v2/summary-private.json'
    previous=read(previous_path)
    if (previous['status']!='rank_win_native_input_encoder_and_single_model_route_verified_not_student_fit'
        or previous['counts']['native_views_checked']!=263
        or not previous['explicit_current_public_event_required_by_native_game_loop']):
        raise ValueError('Actual corrected263-view public source/one-forward proof required')
    files=dict(control['files']);active=control['active_fit_files']
    previous_binding=previous_path.parent/'completion-binding-private.json'
    for p,h in read(previous_binding)['files'].items():
        if p in files and files[p]!=h:raise ValueError('Conflicting immutable source')
        files[p]=h
    for p in [control_path,previous_path,previous_binding]+[ROOT/n for n in PUBLIC_FILES]:
        files[str(p.resolve())]=sha(p)
    if any(sha(p)!=h for p,h in {**files,**active}.items()):raise ValueError('Frozen input changed')
    selected={s['id']:s for s in control['selected']};rows={}
    for spec in control['source_files'].values():
        path=ROOT/spec['path']
        if sha(path)!=spec['sha256']:raise ValueError('Raw source changed')
        for line in path.open():
            row=json.loads(line)
            if row['_meta']['id'] in selected:rows[row['_meta']['id']]=row
    if len(rows)!=118:raise ValueError('All118 broad control views required')
    tokenizer=load_preprocessor(source['training_config']['base_load_path'],backbone_adapter='text')
    args=tomllib.loads(Path(source['training_config']['trainer_config']).read_text())
    proxy=SimpleNamespace(decision_mode='pointer',option_isolation=False)
    counts=Counter();probes={};maximum=0
    for rid,row in rows.items():
        before=copy.deepcopy(row);env,obs=restore(row);event=copy.deepcopy(env.mjai_log[-1])
        request,actions,_=build_win_request(obs,event);public=serving_request(request)
        if serving_request(training_record(row))!=public or row!=before:
            raise ValueError('Full-win serving/training projection changed source or visible input')
        material,mapping=to_record(SystemOneRequest.model_validate(api_request(training_record(row))))
        encoded=DecisionModel.encode(proxy,tokenizer,material,max_state=args['max_state'],max_branch=args['max_branch'],strict=True)
        keys=list(request['questions']['action']['criteria'])
        if encoded['state_truncated'] or len(encoded['ids'])>args['max_packed'] or mapping[0]['keys']!=keys:
            raise ValueError('Full win menu truncates or reorders at actual encoder')
        maximum=max(maximum,len(encoded['ids']))
        for key in keys:
            model=Selector(key);agent=CompleteWinPlacementPolicyAgent(PlacementPolicyClient(model))
            chosen=agent.act(obs,current_public_event=event)
            if (model.calls!=1 or model.last!=public or signature(chosen)!=signature(actions[keys.index(key)])
                or agent.diagnostics['late_full_win_menu_model_calls']!=1):
                raise ValueError('A legal current action failed native one-model execution')
            counts['scripted_all_legal_one_call_choices']+=1
            counts['operation:'+request['questions']['action']['criteria'][key]['type']]+=1
        m=row['_meta'];counts['views']+=1;counts['split:'+m['split']]+=1
        counts['empty_event_views']+=not bool(obs.events)
        kind=next(iter(request['state']['win_context']['candidates'].values()))['kind']
        probes.setdefault((kind,m['placement']['rank']>=3),(obs,event))
        if any(v['type']=='reach' for v in request['questions']['action']['criteria'].values()):
            counts['win_reach:'+m['split']]+=1;probes.setdefault(('reach_control',True),(obs,event))
    if counts['views']!=118 or counts['scripted_all_legal_one_call_choices']!=control['public_menu_arm_budget']//32:
        raise ValueError('Exact118 views and full frozen legal menu coverage required')
    checkpoint=Path(control['continuation_model']['directory']);identity=bundle(checkpoint);model=load_model(checkpoint)
    class Counting:
        def __init__(self):self.calls=0
        def system_one(self,**request):self.calls+=1;return model.system_one(**request)
    for obs,event in probes.values():
        counted=Counting();agent=CompleteWinPlacementPolicyAgent(PlacementPolicyClient(counted))
        chosen=agent.act(obs,current_public_event=event)
        if counted.calls!=1 or signature(chosen) not in {signature(a) for a in obs.legal_actions()}:
            raise ValueError('Actual frozen model failed one-forward legal choice')
        counts['actual_frozen_model_one_call_probes']+=1
    if bundle(checkpoint)!=identity or any(sha(p)!=h for p,h in {**files,**active}.items()):
        raise ValueError('Verification changed weights or immutable inputs')
    out=ROOT/'runs/curriculum-full-win-route-v1/verification-v1';out.mkdir(parents=True,exist_ok=False)
    result=dict(status='actual_complete_current_win_menu_native_encoder_and_single_forward_verified_not_fit',
        input_schema=SCHEMA,policy_method=METHOD,counts=dict(counts),maximum_packed_tokens=maximum,
        exact_same_placement_input_projection=True,explicit_current_public_event_used=True,
        full_current_menu_checked=True,early_and_robbed_kan_shortcuts_unchanged=True,
        heldout_win_reach_coverage=bool(counts['win_reach:validation']),
        heldout_win_reach_absence_not_imputed=True,optimizer_updates=0,
        teacher_and_student_strength_not_established=True,final_grid_unused=True,
        raw_private=True,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'])
    write(out/'summary-private.json',result);write(out/'summary-public.json',result)
    write(out/'completion-binding-private.json',dict(files=files,active_fit_files=active))
    print(json.dumps(result),flush=True)


if __name__=='__main__':verify()

