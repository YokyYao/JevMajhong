"""Acquire fresh held-out late win/RI/ankan/kakan sources without labels."""
from collections import Counter
import copy,json,os,socket,sys,tomllib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_targeted_win_plans_v1 import METHOD,KINDS,plans,digest
from tools.curriculum_targeted_win_source_v1 import construct,restore,SourceUnavailable

PUBLIC_FILES=('tools/curriculum_targeted_win_source_v1.py','tools/curriculum_targeted_win_plans_v1.py',
    'tools/generate_curriculum_targeted_win_sources_v1.py','tests/core/test_curriculum_targeted_win_source_v1.py',
    'configs/training/curriculum-targeted-win-sources-v1.json','cluster/curriculum-targeted-win-sources-v1.slurm',
    'docs/experiments/plans/curriculum-targeted-win-sources-20261123.md')


def generate():
    if (not os.environ.get('SLURM_JOB_ID') or not (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
        or not socket.gethostname().startswith('ws-')):raise RuntimeError('Actual bounded GPU allocation required')
    config=ROOT/'configs/training/curriculum-targeted-win-sources-v1.json';c=read(config)
    if (c['method']!=METHOD or c['kinds']!=list(KINDS) or c['planned_families']!=36 or c['planned_views']!=108
        or c['score_contexts']!=['hold_second','third_gap3000','fourth_gap5000']
        or c['honba']!=0 or c['sticks']!=0 or c['student_fit'] or not c['no_labels']):
        raise ValueError('Fixed fresh36-family/108-view unlabeled recipe required')
    source_path=ROOT/c['source_registration'];source=read(source_path)
    sys.path.insert(0,source['source']);sys.path.insert(0,source['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(1)
    from jevany.api import SystemOneRequest,to_record
    from jevany.data import api_request
    from jevany.model import DecisionModel,load_preprocessor
    from jevmahjong.core.state import signature
    from jevmahjong.training.base_lora import package_digest
    from tools.curriculum_win_context import build_win_request
    from tools.curriculum_endgame_contexts_v2 import score_contexts
    from tools.curriculum_endgame import public_placement
    from tools.curriculum_placement_input_v1 import training_record,serving_request
    from tools.curriculum_placement_full_win_policy_v1 import CompleteWinPlacementPolicyAgent
    from tools.curriculum_placement_input_v1 import PlacementPolicyClient
    from tools.verify_curriculum_rank_win_input_v1 import Selector
    from tools.audit_curriculum_reused_terminal_v1 import fingerprint
    if package_digest(Path(source['source'])/'jevmahjong')!=source['source_sha256']:
        raise ValueError('Pinned public-input runtime changed')
    planned=plans(c)
    if any(2026300000<=p['seed']<2026300032 for p in planned):raise ValueError('Reserved final seed used')
    out=ROOT/c['out'];out.mkdir(parents=True,exist_ok=False)
    active_path=ROOT/'runs/curriculum-progression-v4/launch-binding-private.json';active=read(active_path)['files']
    files={str(p.resolve()):sha(p) for p in [config,source_path,active_path]+[ROOT/n for n in PUBLIC_FILES]}
    for n in PUBLIC_FILES:
        if n.endswith('.py'):compile((ROOT/n).read_text(),str(ROOT/n),'exec')
    write(out/'registered-plans-private.json',dict(method=METHOD,config=c,plans=planned,files=files,
        splits_fixed_before_native_acquisition_or_values=True,labels_generated=0,
        node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']))
    tokenizer=load_preprocessor(source['training_config']['base_load_path'],backbone_adapter='text')
    args=tomllib.loads(Path(source['training_config']['trainer_config']).read_text())
    proxy=SimpleNamespace(decision_mode='pointer',option_isolation=False)
    pools={s:[] for s in ('train','validation')};quarantine=[];counts=Counter();fps={s:set() for s in pools};maximum=0
    for number,plan in enumerate(planned,1):
        fresh=[]
        try:
            for context in c['score_contexts']:
                scores=score_contexts(3,0)[context]['scores']
                env,obs,native=construct(plan['kind'],plan['target14'],plan['seed'],scores)
                request,actions,_=build_win_request(obs,env.mjai_log[-1]);q=request['questions']['action'];keys=list(q['criteria'])
                if len(keys)>16:raise ValueError('Native target menu exceeds the predeclared cap')
                expected={'win_reach':'reach','win_ankan':'ankan','win_kakan':'kakan'}[plan['kind']]
                if expected not in {v['type'] for v in q['criteria'].values()}:raise ValueError('Native special opportunity missing')
                physical='targeted-win:'+plan['canonical_shape_family']+':'+str(plan['seed'])
                row=dict(request,_meta=dict(id=physical+':'+context,physical_id=physical,group_id=plan['group_id'],
                    split=plan['split'],category='win',score_context=context,placement=public_placement(request['state']),
                    targeted_operation=plan['kind'],canonical_shape_family=plan['canonical_shape_family'],
                    native_source=native,source_actions_constructed_not_expert_labels=True,
                    targeted_not_natural_game_distribution=True,label_type='unlabelled_native_targeted_win',labels_generated=False))
                fork,other=restore(row)
                if build_win_request(other,fork.mjai_log[-1])[0]!=request:raise ValueError('Actual targeted source replay differs')
                for key in (next(k for k,v in q['criteria'].items() if v['type']=='hora'),
                    next(k for k,v in q['criteria'].items() if v['type']==expected)):
                    model=Selector(key);agent=CompleteWinPlacementPolicyAgent(PlacementPolicyClient(model))
                    action=agent.act(obs,current_public_event=env.mjai_log[-1])
                    if model.calls!=1 or signature(action)!=signature(actions[keys.index(key)]):
                        raise ValueError('Actual native targeted action failed one-forward route')
                    counts['scripted_native_one_call_checks']+=1
                record,mapping=to_record(SystemOneRequest.model_validate(api_request(training_record(row))))
                encoded=DecisionModel.encode(proxy,tokenizer,record,max_state=args['max_state'],max_branch=args['max_branch'],strict=True)
                if encoded['state_truncated'] or len(encoded['ids'])>args['max_packed'] or mapping[0]['keys']!=keys:
                    raise ValueError('Actual targeted input truncates or reorders the native menu')
                maximum=max(maximum,len(encoded['ids']));fresh.append(row)
        except SourceUnavailable as error:
            quarantine.append(dict(plan=plan,reason=str(error),no_labels_generated=True,no_replacement_after_values=True))
            counts['source_families_quarantined']+=1
            continue
        if len(fresh)!=3:raise ValueError('All matched score contexts required per admitted source family')
        for row in fresh:
            split=row['_meta']['split'];public=serving_request(row)
            fps[split].update((fingerprint(public),fingerprint(public,True)))
            pools[split].append(row);counts[split+':'+plan['kind']]+=1
        counts['native_families_admitted']+=1
        if number%6==0:print(json.dumps(dict(stage='targeted_win_source',families_processed=number,
            families_admitted=counts['native_families_admitted'],quarantined=counts['source_families_quarantined'])),flush=True)
    if fps['train'] & fps['validation']:raise ValueError('Cross-split targeted public input overlap')
    replay_path=ROOT/'runs/curriculum-yaku-scenarios-v1/data-v2/data-ready-private.json';replay=read(replay_path)
    files[str(replay_path.resolve())]=sha(replay_path)
    for split in pools:
        path=ROOT/replay['files'][split]['path'];files[str(path.resolve())]=sha(path)
        if files[str(path.resolve())]!=replay['files'][split]['sha256']:raise ValueError('Current3B replay changed')
        other='validation' if split=='train' else 'train';other_groups={r['_meta']['group_id'] for r in pools[other]}
        for line in path.open():
            row=json.loads(line);public=serving_request(row)
            if row['_meta']['group_id'] in other_groups or {fingerprint(public),fingerprint(public,True)} & fps[other]:
                raise ValueError('Targeted source overlaps opposite current replay input/group')
    outputs={};families={};operation_families={}
    for split,rows in pools.items():
        path=out/f'{split}-targeted-win-unlabelled-private.jsonl'
        with path.open('x') as f:
            for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
        families[split]=len({r['_meta']['group_id'] for r in rows})
        operation_families[split]={kind:len({r['_meta']['group_id'] for r in rows if r['_meta']['targeted_operation']==kind}) for kind in KINDS}
        outputs[split]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),records=len(rows),groups=families[split])
    if any(operation_families['validation'][k]<2 for k in KINDS):
        raise ValueError('Insufficient actual held-out native operation families; no coverage imputed')
    if any(sha(p)!=h for p,h in {**files,**active}.items()):raise ValueError('Bound source or active training input changed')
    result=dict(status='fresh_targeted_late_win_special_native_sources_prepared_no_labels',
        method=METHOD,files=outputs,counts=dict(counts),families=families,operation_families=operation_families,
        canonical_families_split_before_values=True,maximum_packed=maximum,source_quarantine=len(quarantine),
        all_native_sources_reconstructed=True,heldout_RI_ankan_kakan_win_windows_actual=True,
        exact_public_input_and_group_opposite_3B_overlap=0,original_replay_untouched=True,
        labels_generated=0,student_updates=0,encoder_default_label_never_used_as_supervision=True,
        conditional_constructed_distribution_not_natural_strength_test=True,raw_private=True,final_grid_unused=True,
        node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'])
    write(out/'source-quarantine-private.json',quarantine)
    write(out/'data-ready-private.json',result);write(out/'completion-binding-private.json',dict(files=files,active_fit_files=active))
    write(out/'summary-public.json',{k:v for k,v in result.items() if k!='files'})
    print(json.dumps(read(out/'summary-public.json')),flush=True)

if __name__=='__main__':generate()

