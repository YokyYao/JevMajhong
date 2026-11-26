"""Project all late curriculum rows into one declared placement objective."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import socket
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only,write
from tools.curriculum_transfer import read,sha
from tools.curriculum_placement_input_v1 import serving_request,training_record,SCHEMA
from tools.curriculum_endgame import in_endgame_scope


def prepare():
    gpu_only();reg_path=ROOT/'runs/curriculum-progression-v1/fit-plan/registered.json';reg=read(reg_path)
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from types import SimpleNamespace
    from jevany.api import SystemOneRequest,to_record
    from jevany.data import api_request
    from jevany.model import DecisionModel,load_preprocessor
    base_path=ROOT/'runs/curriculum-rank-course-data-v1/data-v1/data-ready-private.json';base=read(base_path)
    new_path=ROOT/'runs/curriculum-rank-nonwin-labels-v1/merged-v1/data-ready-private.json';new=read(new_path)
    proof_path=ROOT/'runs/curriculum-placement-input-v1/verification-v1/summary-private.json';proof=read(proof_path)
    if (base['status']!='exact3B_replay_plus_rank_win_input_contract_data_prepared_not_fit_admitted'
        or new['status']!='actual_rank32_nonwin_soft_targets_generated_all_late_input_and_fit_admission_pending'
        or proof['status']!='all_late_placement_instruction_encoder_and_early_parity_verified_not_fit_admitted'
        or proof['input_schema']!=SCHEMA or proof['optimizer_steps']!=0):
        raise ValueError('Actual completed labels and all-late input verification required')
    files={}
    for prerequisite in (base_path,new_path,proof_path):
        binding=prerequisite.parent/'completion-binding-private.json'
        for p,h in read(binding)['files'].items():
            if p in files and files[p]!=h:raise ValueError('Conflicting immutable ancestry')
            if sha(p)!=h:raise ValueError('Prerequisite changed: '+p)
            files[p]=h
        files[str(binding.resolve())]=sha(binding);files[str(prerequisite.resolve())]=sha(prerequisite)
    for path in (Path(__file__),reg_path,ROOT/'cluster/curriculum-placement-data-v1.slurm'):
        files[str(path.resolve())]=sha(path)
    out=ROOT/'runs/curriculum-placement-data-v1/data-v1';out.mkdir(parents=True,exist_ok=False)
    tokenizer=load_preprocessor(reg['training_config']['base_load_path'],backbone_adapter='text')
    args=tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    proxy=SimpleNamespace(decision_mode='pointer',option_isolation=False)
    ids=set();groups={};outputs={};counts={};sources={};maximum=0
    for split in ('train','validation'):
        candidates=[];original_count=0;new_count=0;changed=0;early=0;kept_base_win=0
        for parent,is_new in ((base,False),(new,True)):
            path=ROOT/parent['files'][split]['path'];digest=sha(path)
            if digest!=parent['files'][split]['sha256']:raise ValueError('Candidate data changed')
            files[str(path.resolve())]=digest
            for line in path.open():
                row=json.loads(line);before=copy.deepcopy(row);projected=training_record(row)
                a,b=row['questions']['action'],projected['questions']['action']
                if (row!=before or projected['state']!=row['state'] or projected['_meta']!=row['_meta']
                    or any(a.get(k)!=b.get(k) for k in ('type','criteria','label','target','src'))
                    or serving_request(projected)!=serving_request(row)):
                    raise ValueError('Only declared instructions may change; supervision/lineage must remain exact')
                late=in_endgame_scope(row['state']);different=a['instructions']!=b['instructions']
                if not late and different:raise ValueError('Early replay instruction changed')
                changed+=different;early+=not late
                if is_new and not a['src'].startswith('curriculum4:rank_nonwin:'):raise ValueError('Unexpected appended source')
                if not is_new:
                    original_count+=not a['src'].startswith('curriculum4:rank_win:')
                    kept_base_win+=a['src'].startswith('curriculum4:rank_win:')
                else:new_count+=1
                # Unchanged rows reuse the actual earlier admission proof.
                # Every changed instruction and appended source is encoded now.
                if different or is_new:
                    record,mapping=to_record(SystemOneRequest.model_validate(api_request(projected)))
                    encoded=DecisionModel.encode(proxy,tokenizer,record,max_state=args['max_state'],
                        max_branch=args['max_branch'],strict=True)
                    if (encoded['state_truncated'] or len(encoded['ids'])>args['max_packed']
                        or mapping[0]['keys']!=list(b['criteria'])):raise ValueError('New projection truncates/reorders input')
                    maximum=max(maximum,len(encoded['ids']))
                if projected['_meta']['id'] in ids:raise ValueError('Duplicate/cross-split ID')
                ids.add(projected['_meta']['id']);candidates.append(projected)
        if original_count!={'train':91247,'validation':24516}[split]:raise ValueError('All original3B replay required')
        groups[split]={r['_meta']['group_id'] for r in candidates}
        sources[split]=dict(Counter(r['questions']['action']['src'] for r in candidates))
        dest=out/f'{split}-all-late-placement-private.jsonl'
        with dest.open('x') as stream:
            for row in candidates:stream.write(json.dumps(row,separators=(',',':'))+'\n')
        outputs[split]=dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),records=len(candidates),groups=len(groups[split]))
        counts[split]=dict(original3B=original_count,existing_rank_win=kept_base_win,new_rank_nonwin=new_count,
            late_instruction_changes=changed,early_rows_unchanged=early,total=len(candidates))
    if groups['train']&groups['validation']:raise ValueError('Source-family split overlap')
    if any(sha(p)!=h for p,h in files.items()):raise ValueError('Frozen prerequisite changed')
    result=dict(status='all_late_placement_replay_win_and_nonwin_data_prepared_not_fit_admitted',
        input_schema=SCHEMA,files=outputs,counts=counts,sources=sources,maximum_changed_packed=maximum,
        original_states_menus_labels_targets_groups_preserved=True,
        late_instructions_intentionally_changed=True,early_input_objects_unchanged=True,
        one_placement_objective_for_all_late_decisions=True,masked_tested_action_target_loss_required=True,
        no_unlabelled_option_negative_targets=True,source_group_overlap=0,optimizer_steps=0,
        teacher_uncertainty_and_quarantine_preserved=True,mix_parent_budget_not_registered=True,
        raw_data_private=True,final_grid_unused=True,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'])
    write(out/'data-ready-private.json',result);write(out/'completion-binding-private.json',dict(files=files))
    write(out/'summary-public.json',{k:v for k,v in result.items() if k not in ('files','sources')})
    print(json.dumps(read(out/'summary-public.json')),flush=True)


if __name__=='__main__':prepare()
