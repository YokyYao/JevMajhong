"""Append complete-current-menu win controls to coherent placement replay."""
from collections import Counter
import copy,json,os,socket,sys,tomllib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only,write
from tools.curriculum_transfer import read,sha
from tools.curriculum_placement_input_v1 import training_record,serving_request,SCHEMA
from tools.curriculum_placement_full_win_policy_v1 import METHOD

def prepare():
    gpu_only();source_path=ROOT/'runs/curriculum-progression-v1/fit-plan/registered.json';source=read(source_path)
    sys.path.insert(0,source['source']);sys.path.insert(0,source['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevany.api import SystemOneRequest,to_record
    from jevany.data import api_request
    from jevany.model import DecisionModel,load_preprocessor
    base_path=ROOT/'runs/curriculum-placement-data-v1/data-v1/data-ready-private.json';base=read(base_path)
    control_path=ROOT/'runs/curriculum-rank-win-controls-v1/merged-v1/data-ready-private.json';control=read(control_path)
    proof_path=ROOT/'runs/curriculum-full-win-route-v1/verification-v1/summary-private.json';proof=read(proof_path)
    if (base['status']!='all_late_placement_replay_win_and_nonwin_data_prepared_not_fit_admitted'
        or control['status']!='actual_rank32_full_menu_win_control_targets_generated_route_and_fit_pending'
        or proof['status']!='actual_complete_current_win_menu_native_encoder_and_single_forward_verified_not_fit'
        or proof['policy_method']!=METHOD or proof['input_schema']!=SCHEMA or not proof['full_current_menu_checked']):
        raise ValueError('Actual completed data and full-menu native route proof required')
    files={}
    for path in (base_path,control_path,proof_path):
        binding=path.parent/'completion-binding-private.json'
        for p,h in read(binding)['files'].items():
            if p in files and files[p]!=h:raise ValueError('Conflicting immutable ancestry')
            if sha(p)!=h:raise ValueError('Completed prerequisite changed')
            files[p]=h
        files[str(path.resolve())]=sha(path);files[str(binding.resolve())]=sha(binding)
    for path in (Path(__file__),source_path,ROOT/'cluster/curriculum-placement-data-v2.slurm',
        ROOT/'tools/curriculum_placement_objective_v1.py',ROOT/'tests/core/test_curriculum_placement_objective_v1.py'):
        files[str(path.resolve())]=sha(path)
    tokenizer=load_preprocessor(source['training_config']['base_load_path'],backbone_adapter='text')
    args=tomllib.loads(Path(source['training_config']['trainer_config']).read_text())
    proxy=SimpleNamespace(decision_mode='pointer',option_isolation=False)
    out=ROOT/'runs/curriculum-placement-data-v2/data-v1';out.mkdir(parents=True,exist_ok=False)
    ids=set();groups={};outputs={};counts={};sources={};group_counts={};maximum=0
    for split in ('train','validation'):
        rows=[];kept=0;added=0;source_groups={}
        for parent,is_new in ((base,False),(control,True)):
            path=ROOT/parent['files'][split]['path']
            if sha(path)!=parent['files'][split]['sha256']:raise ValueError('Prepared labels changed')
            files[str(path.resolve())]=sha(path)
            for line in path.open():
                row=json.loads(line);original=copy.deepcopy(row);projected=training_record(row);q=projected['questions']['action']
                if row!=original or projected['_meta']!=row['_meta'] or serving_request(row)!=serving_request(projected):
                    raise ValueError('Projection mutated lineage or serving surface')
                if any(q.get(k)!=row['questions']['action'].get(k) for k in ('label','target','src','criteria')):
                    raise ValueError('Supervision changed')
                if not is_new:
                    if projected!=row:raise ValueError('Already coherent replay must remain object-exact')
                    kept+=1
                else:
                    if (not q['src'].startswith('curriculum4:rank_win:control:') or any(v<=0 for v in q['target'].values())
                        or set(q['target'])!=set(q['criteria'])):raise ValueError('Complete current-action control target required')
                    material,mapping=to_record(SystemOneRequest.model_validate(api_request(projected)))
                    encoded=DecisionModel.encode(proxy,tokenizer,material,max_state=args['max_state'],max_branch=args['max_branch'],strict=True)
                    if encoded['state_truncated'] or len(encoded['ids'])>args['max_packed'] or mapping[0]['keys']!=list(q['criteria']):
                        raise ValueError('Control input truncates or reorders menu')
                    maximum=max(maximum,len(encoded['ids']));added+=1
                rid=projected['_meta']['id']
                if rid in ids:raise ValueError('Duplicate/cross-split ID')
                ids.add(rid);rows.append(projected)
                source_groups.setdefault(q['src'],set()).add(projected['_meta']['group_id'])
        if kept!=base['files'][split]['records'] or added!=control['files'][split]['records']:
            raise ValueError('Every original and admitted control row required')
        groups[split]={r['_meta']['group_id'] for r in rows}
        sources[split]=dict(Counter(r['questions']['action']['src'] for r in rows))
        group_counts[split]={s:len(v) for s,v in source_groups.items()}
        path=out/f'{split}-placement-all-sources-private.jsonl'
        with path.open('x') as stream:
            for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
        outputs[split]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),records=len(rows),groups=len(groups[split]))
        counts[split]=dict(existing_placement_rows_exact=kept,new_full_menu_win_controls=added,total=len(rows))
    if groups['train'] & groups['validation']:raise ValueError('Source families cross splits')
    if any(sha(p)!=h for p,h in files.items()):raise ValueError('Immutable input changed')
    result=dict(status='placement_replay_partial_win_full_win_controls_and_nonwin_prepared_not_fit',
        input_schema=SCHEMA,required_policy_method=METHOD,files=outputs,counts=counts,
        source_records=sources,source_distinct_families=group_counts,maximum_new_packed=maximum,
        all_existing_coherent_replay_objects_exact=True,all_source_labels_targets_groups_preserved=True,
        source_group_overlap=0,uncertain_labels_and_quarantine_preserved=True,
        partial_targets_require_tested_action_mask=True,full_RI_soft_class_loss_required=True,
        actual_exposure_mix_and_repeat_accounting_still_to_register=True,
        win_reach_heldout_native_coverage=proof['heldout_win_reach_coverage'],
        win_kan_native_coverage=any(proof['counts'].get('operation:'+op,0) for op in ('ankan','kakan','daiminkan')),
        coverage_gaps_not_imputed=True,parent_budget_and_optimizer_not_registered=True,
        optimizer_updates=0,raw_private=True,final_grid_unused=True,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'])
    write(out/'data-ready-private.json',result);write(out/'completion-binding-private.json',dict(files=files))
    public={k:v for k,v in result.items() if k not in ('files','source_records','source_distinct_families')}
    write(out/'summary-public.json',public);print(json.dumps(public),flush=True)

if __name__=='__main__':prepare()

