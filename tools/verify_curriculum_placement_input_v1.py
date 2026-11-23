"""Verify coherent placement instructions using prior native and win-route evidence."""
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only,write
from tools.curriculum_transfer import read,sha
from tools.curriculum_placement_input_v1 import serving_request,training_record,SCHEMA,METHOD,INSTRUCTION

PUBLIC_FILES=('tools/curriculum_placement_input_v1.py','tools/verify_curriculum_placement_input_v1.py',
    'tests/core/test_curriculum_placement_input_v1.py','cluster/curriculum-placement-input-v1.slurm',
    'docs/experiments/plans/curriculum-placement-input-20261123.md',
    'tools/prepare_curriculum_placement_data_v1.py','cluster/curriculum-placement-data-v1.slurm')


def verify():
    gpu_only(); reg_path=ROOT/'runs/curriculum-progression-v1/fit-plan/registered.json';reg=read(reg_path)
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from types import SimpleNamespace
    from jevmahjong.policy import unified
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.policy.routing import load_model
    from jevmahjong.evaluation.unified_contract import bundle
    from jevany.api import SystemOneRequest,to_record,validate_distribution
    from jevany.data import api_request
    from jevany.model import DecisionModel,load_preprocessor
    from tools.curriculum_rank_win_input_v1 import serving_request as previous,SCHEMA as WIN_SCHEMA
    from tools.curriculum_endgame import in_endgame_scope,public_placement
    proof_path=ROOT/'runs/curriculum-rank-win-input-v1/verification-v1/summary-private.json';proof=read(proof_path)
    if (proof['status']!='rank_win_native_input_encoder_and_single_model_route_verified_not_student_fit'
        or proof['input_schema']!=WIN_SCHEMA or not proof['same_model_and_parameter_bundle']
        or proof['counts']['native_views_checked']!=263 or not proof['no_optimizer_updates']):
        raise ValueError('Actual native win-source/single-call proof required before extension')
    binding_path=proof_path.parent/'completion-binding-private.json';files=dict(read(binding_path)['files'])
    if any(sha(p)!=h for p,h in files.items()):raise ValueError('Prior win-route evidence inputs changed')
    if (not Path(unified.__file__).resolve().is_relative_to(Path(reg['source']).resolve())
        or package_digest(Path(reg['source'])/'jevmahjong')!=reg['source_sha256']):
        raise ValueError('Pinned runtime changed')
    raw_ready_path=ROOT/'runs/curriculum-reused-terminal-v1/collection-v1/summary-private.json'
    raw_ready=read(raw_ready_path)
    paths=[reg_path,proof_path,binding_path,raw_ready_path]+[ROOT/name for name in PUBLIC_FILES]
    for name in PUBLIC_FILES:
        if name.endswith('.py'):compile((ROOT/name).read_text(),str(ROOT/name),'exec')
    out=ROOT/'runs/curriculum-placement-input-v1/verification-v1';out.mkdir(parents=True,exist_ok=False)
    tokenizer=load_preprocessor(reg['training_config']['base_load_path'],backbone_adapter='text')
    recipe=tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    proxy=SimpleNamespace(decision_mode='pointer',option_isolation=False)
    counts=Counter();maximum=0;probes={};early_probes=[]
    for split in ('train','validation'):
        path=ROOT/raw_ready['files'][split]['path'];paths.append(path)
        if sha(path)!=raw_ready['files'][split]['sha256']:raise ValueError('Original source corpus changed')
        for line in path.open():
            row=json.loads(line);old=previous(row);new=serving_request(row);record=training_record(row)
            if (old['state']!=new['state'] or old['questions']['action']['criteria']!=new['questions']['action']['criteria']
                or new['questions']['action']['type']!=old['questions']['action']['type']
                or serving_request(record)!=new or not new['questions']['action']['instructions'].startswith(INSTRUCTION)):
                raise ValueError('Only the declared late instruction may differ')
            material,mapping=to_record(SystemOneRequest.model_validate(api_request(record)))
            encoded=DecisionModel.encode(proxy,tokenizer,material,max_state=recipe['max_state'],
                max_branch=recipe['max_branch'],strict=True)
            if (encoded['state_truncated'] or len(encoded['ids'])>recipe['max_packed']
                or mapping[0]['keys']!=list(new['questions']['action']['criteria'])):
                raise ValueError('Placement input truncates or reorders actions')
            maximum=max(maximum,len(encoded['ids']));counts['raw_views_encoded']+=1
            category=row['_meta']['category'];counts['category:'+category]+=1
            behind=public_placement(new['state'])['rank']>=3
            if category=='win':
                kind=next(iter(new['state']['win_context']['candidates'].values()))['kind']
                bucket=(category,kind,behind)
            else:bucket=(category,behind)
            probes.setdefault(bucket,row)
    if counts['raw_views_encoded']!=4580 or len(probes)!=10:
        raise ValueError('All4580 inputs and ten win/non-win placement buckets required')
    data_path=ROOT/'runs/curriculum-yaku-scenarios-v1/data-v2/data-ready-private.json';data=read(data_path);paths.append(data_path)
    path=ROOT/data['files']['validation']['path'];paths.append(path)
    if sha(path)!=data['files']['validation']['sha256']:raise ValueError('Original validation changed')
    for line in path.open():
        row=json.loads(line);old=previous(row);new=serving_request(row)
        if in_endgame_scope(new['state']):
            if (new['state']!=old['state'] or new['questions']['action']['criteria']!=old['questions']['action']['criteria']
                or not new['questions']['action']['instructions'].startswith(INSTRUCTION)):
                raise ValueError('Late replay changed outside the declared instruction')
            counts['late_replay_instruction_changes']+=new['questions']['action']['instructions']!=old['questions']['action']['instructions']
            if any(v['type']=='hora' for v in new['questions']['action']['criteria'].values()) and 'win_context' not in new['state']:
                counts['legacy_late_win_without_source_no_payer_inferred']+=1
        else:
            if new!=old:raise ValueError('Early replay input changed')
            counts['early_replay_inputs_identical']+=1
            if len(early_probes)<2:early_probes.append(row)
        counts['old_validation_rows_checked']+=1
    if counts['old_validation_rows_checked']!=24516 or not early_probes:
        raise ValueError('All current validation rows and actual early replay required')
    model_path=Path(reg['curriculum']['reference_checkpoint']);identity=bundle(model_path)
    if identity['identity_sha256']!=reg['curriculum']['reference_identity']:raise ValueError('Frozen v8 changed')
    model=load_model(model_path)
    for row in probes.values():
        public=serving_request(row);result=model.system_one(**public)
        validate_distribution(result['answers']['action']['probabilities'],list(public['questions']['action']['criteria']))
        counts['actual_late_forward_probes']+=1
    error=0.
    for row in early_probes:
        old=model.system_one(**previous(row));new=model.system_one(**serving_request(row))
        a=old['answers']['action']['probabilities'];b=new['answers']['action']['probabilities']
        error=max(error,max(abs(a[k]-b[k]) for k in a));counts['actual_early_paired_probes']+=1
    if error>=5e-5 or bundle(model_path)!=identity:raise ValueError('Early parity or weight immutability failed')
    for path in paths:
        key=str(path.resolve());digest=sha(path)
        if key in files and files[key]!=digest:raise ValueError('Conflicting bound input identity')
        files[key]=digest
    active=read(ROOT/'runs/curriculum-progression-v1/launch-binding-private.json')['files']
    if any(sha(p)!=h for p,h in {**files,**active}.items()):raise ValueError('A frozen source changed')
    result=dict(status='all_late_placement_instruction_encoder_and_early_parity_verified_not_fit_admitted',
        input_schema=SCHEMA,policy_method=METHOD,counts=dict(counts),maximum_packed=maximum,
        inherited_actual_win_source_single_call_proof=True,early_forward_max_probability_error=error,
        late_parent_predictions_intentionally_not_equated=True,all_late_windows_share_one_placement_objective=True,
        same_model_parameters_and_execution_route=True,optimizer_steps=0,
        forward_probes_not_strength_or_student_validation=True,parent_mix_loss_fit_pending=True,
        main_Windows_and_frozen_v8_unchanged=True,final_grid_unused=True,
        node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'])
    write(out/'summary-private.json',result);write(out/'summary-public.json',result)
    write(out/'completion-binding-private.json',dict(files=files,active_fit_files=active))
    print(json.dumps(result),flush=True)


if __name__=='__main__':verify()
