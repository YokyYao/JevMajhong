"""Seal a qualified basic final for twelve-source calls, retaining real exits."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_parent import (ParentNotReady,read,rows,sha,require_stage1_candidate,
    require_child_input_contract,require_replay)
from tools.seal_curriculum_basic_repair import require_source_quality
from tools.curriculum_transfer import METHOD


def scheduler_exit(receipt,raw,expected_command=None):
    values=dict(token.split('=',1) for token in raw.split() if '=' in token)
    if (receipt.get('exit_code')!=0 or str(receipt.get('job_id'))!=values.get('JobId')
        or values.get('JobState')!='COMPLETED' or values.get('ExitCode')!='0:0'
        or not values.get('NodeList') or values['NodeList'].startswith(('lo-','login'))
        or 'gres/gpu=' not in values.get('AllocTRES','')
        or any(values.get(k) in (None,'Unknown','N/A','(null)') for k in ('StartTime','EndTime'))):
        raise ParentNotReady('Actual successful terminal allocated-GPU scheduler evidence required')
    gpu=values['AllocTRES'].split('gres/gpu=',1)[1].split(',',1)[0]
    if not gpu.isdigit() or int(gpu)<1:
        raise ParentNotReady('CPU-only or zero-GPU scheduler exit is not admissible')
    if expected_command is not None and values.get('Command')!=str(Path(expected_command).resolve()):
        raise ParentNotReady('Scheduler receipt belongs to a different submitted entry')
    return dict(job_id=str(receipt['job_id']),status=raw.strip())


def seal(registered,evidence,fit_exit,audit_exit,child_data_ready,out,fit_archive=None,audit_archive=None,capture_binding=None):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-','login'))):
        raise ParentNotReady('A bounded scheduler GPU allocation is required')
    registered=Path(registered).resolve();out=Path(out).resolve()
    if out.exists():raise ParentNotReady('Never overwrite a parent seal')
    reg=read(registered);c=reg['curriculum'];fit=ROOT/c['out'];run=registered.parent.parent
    completion=read(fit/'completion.json');audit=read(evidence);validation=rows(fit/'metrics/validation_metrics.jsonl')[-1]
    quality=require_source_quality(reg,audit,validation)
    reloads=[r for r in audit.get('saved_reloads',[]) if r['step']==reg['registered_final_step']]
    identity=reloads[0]['identity'] if len(reloads)==1 else None
    require_stage1_candidate(reg,completion,audit,validation,identity)
    if c.get('riichi_input_schema') is not None:
        raise ParentNotReady('This version requires the prospective plain basic-repair contract')
    files=dict(reg['inputs']);scheduler=[]
    for role,exit_file,archive in (('fit',fit_exit,fit_archive),('audit',audit_exit,audit_archive)):
        receipt=read(exit_file);expected_command=run/(role+'.slurm')
        if archive is not None:
            archive=Path(archive).resolve()
            if archive!=run/(role+'-terminal-scheduler-private.json'):
                raise ParentNotReady('Use the actual managed parent terminal archive')
            captured=read(archive)
            if captured.get('exit')!=receipt:raise ParentNotReady('Archived/actual exit receipts differ')
            raw=captured['raw'];files[str(archive)]=sha(archive)
            # The fit archive is emitted by the launch-bound audit; the audit
            # archive is emitted by the preflight-bound post-audit task.
            capture=run/('audit-private.py' if role=='fit' else 'post-audit-private.py')
            if role=='audit':
                if capture_binding is None:raise ParentNotReady('Before-completion terminal-capture binding required')
                capture_binding=Path(capture_binding).resolve();prerequisite=read(capture_binding)
                observed=dict(t.split('=',1) for t in prerequisite.get('observed_parent_scheduler','').split() if '=' in t)
                if (prerequisite.get('status')!='basic_parent_terminal_capture_bound_before_completion'
                    or prerequisite.get('parent_registration')!=str(registered)
                    or prerequisite.get('parent_registration_sha256')!=sha(registered)
                    or not 0<=prerequisite.get('observed_step',-1)<reg['registered_final_step']
                    or observed.get('JobState')!='RUNNING' or observed.get('JobId')!=str(read(fit_exit)['job_id'])):
                    raise ParentNotReady('Capture code binding was not observed on the running parent before completion')
                files[str(capture_binding)]=sha(capture_binding)
            else:prerequisite=read(run/'launch-binding-private.json')
            bound=prerequisite.get('files',prerequisite)
            for p,h in bound.items():
                if sha(p)!=h:raise ParentNotReady('Terminal capture binding changed: '+p)
                files[p]=h
            if bound.get(str(capture.resolve()))!=sha(capture):
                raise ParentNotReady('Terminal capture implementation was not bound before completion')
            files[str(capture.resolve())]=sha(capture)
        else:
            p=subprocess.run(['scontrol','-o','show','job',str(int(receipt['job_id']))],check=True,
                capture_output=True,text=True,timeout=20)
            raw=p.stdout
        scheduler.append(scheduler_exit(receipt,raw,expected_command))
        files[str(Path(exit_file).resolve())]=sha(exit_file)
    for binding_name in ('launch-binding-private.json','preflight-binding-private.json'):
        path=run/binding_name;binding=read(path);bound=binding.get('files',binding)
        for p,h in bound.items():
            if sha(p)!=h:raise ParentNotReady('Parent managed-launch/audit input changed: '+p)
            if p in files and files[p]!=h:raise ParentNotReady('Conflicting sealed input')
            files[p]=h
        files[str(path.resolve())]=sha(path)
    if any(sha(p)!=h for p,h in files.items()):raise ParentNotReady('Registered parent input changed')
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    if (package_digest(Path(reg['source'])/'jevmahjong')!=reg['source_sha256']
        or package_digest(Path(reg['training_config']['upstream_source'])/'jevany')!=reg['training_config']['upstream_source_sha256']):
        raise ParentNotReady('Pinned package changed')
    checkpoint=fit/'training-run/checkpoints'/f"step-{reg['registered_final_step']:06d}";model=bundle(checkpoint)
    require_stage1_candidate(reg,completion,audit,validation,model['identity_sha256'])
    args=tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    if args['max_steps']!=reg['registered_final_step'] or args.get('early_stop_patience',0)!=0:
        raise ParentNotReady('Fixed complete parent budget required')
    child=read(child_data_ready);require_child_input_contract(reg,child)
    if (child.get('status')!='calls_repair12_source_data_prepared_not_parent_or_fit_admitted'
        or child.get('source_specific_basic_parent_seal_required') is not True):
        raise ParentNotReady('Audited twelve-source calls data required')
    for spec in child['files'].values():
        if sha(ROOT/spec['path'])!=spec['sha256']:raise ParentNotReady('Child data changed')
    parent_train=rows(args['data']);parent_val=rows(args['eval_data'])
    child_train=rows(ROOT/child['files']['train']['path']);child_val=rows(ROOT/child['files']['validation']['path'])
    require_replay(parent_train,child_train);require_replay(parent_val,child_val)
    groups=lambda pool:{r['_meta']['group_id'] for r in pool}
    reserved=ROOT/c['data_root']/'reserved-public-groups.json'
    held=groups(parent_val)|groups(child_val)|set(read(reserved))
    if groups(child_train)&held or groups(parent_train)&groups(child_val):raise ParentNotReady('Held-out/reserved overlap')
    paths=[registered,Path(evidence),Path(child_data_ready),reserved,fit/'completion.json',fit/'protocol.json',
        fit/'initialization-audit.json',fit/'actual-update-proof.json',Path(__file__),ROOT/'tools/curriculum_parent.py',
        ROOT/'tools/seal_curriculum_basic_repair.py',ROOT/'tools/curriculum_basic_repair.py',
        ROOT/'docs/experiments/plans/curriculum-calls-repair-fit-20261122.md']
    paths.extend(fit/'metrics'/name for name in ('train_metrics.jsonl','validation_metrics.jsonl','evaluation_metrics.jsonl'))
    paths.extend(ROOT/spec['path'] for spec in child['files'].values())
    for step in completion['curve_steps']:
        paths.extend(fit/'evaluations'/f'step-{step:06d}'/name for name in ('protocol.json','summary.json','matches-private.jsonl'))
    files.update({str(p.resolve()):sha(p) for p in paths})
    files.update({str((checkpoint/name).resolve()):h for name,h in model['files'].items()})
    result=dict(status='curriculum_stage1_final_parent_sealed',method=METHOD,node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'],model=model,selected_step=reg['registered_final_step'],
        selection_from_minimum_CE=False,optimizer_resumed=False,parent_registration=str(registered),
        parent_evidence=str(Path(evidence).resolve()),child_data_ready=str(Path(child_data_ready).resolve()),
        required_input_schema=None,parent_training_file=str(Path(args['data']).resolve()),
        ancestor_training_files=[str(Path(args['data']).resolve())],parent_validation_files=[str(Path(args['eval_data']).resolve())],
        bound_files=files,stage_quality=audit['stage1_advancement'],original_stage1_advancement=audit['stage1_advancement'],
        scheduler_exits=scheduler,source_specific_parent_quality_checked=True,basic_repair_source_quality=quality,
        source_specific_sealer='tools/seal_curriculum_basic_repair_v2.py',base=c['base'],base_revision=c['base_revision'],
        group_proof=dict(parent_train_groups=len(groups(parent_train)),child_train_groups=len(groups(child_train)),
            child_validation_groups=len(groups(child_val)),overlap_with_any_held_group=0,
            mandatory_stage1_replay_present=True,mandatory_stage1_validation_replay_present=True,
            replay_public_inputs_labels_targets_and_groups_equal=True),child_fit_started=False,practical_v8_unchanged=True)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('registered','evidence','fit-exit','audit-exit','child-data-ready','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('fit-archive','audit-archive','capture-binding'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();seal(a.registered,a.evidence,a.fit_exit,a.audit_exit,a.child_data_ready,a.out,a.fit_archive,a.audit_archive,a.capture_binding)
