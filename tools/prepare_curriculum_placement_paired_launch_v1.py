"""Bind the actual paired-parent integration and unchanged placement launch."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only,write
from tools.curriculum_placement_runtime_v1 import require_registration
from tools.curriculum_placement_parent_v1 import verify_parent
from tools.curriculum_placement_assessment_v1 import require_criteria,early_profile
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
from tools.capture_curriculum_yaku_terminal_v1 import accounting_proof

PUBLIC_FILES=(
    'tools/prepare_curriculum_placement_paired_launch_v1.py','tools/curriculum_paired_placement_launch_v1.py',
    'cluster/curriculum-placement-paired-v1.slurm','configs/training/curriculum-placement-paired-v1.json',
    'tests/core/test_curriculum_paired_placement_launch_v1.py','docs/experiments/plans/curriculum-paired-placement-continuation-20261124.md',
    'tools/audit_curriculum_placement_course_v1.py','tools/curriculum_placement_assessment_v1.py',
    'configs/training/curriculum-placement-assessment-v1.json','tests/core/test_curriculum_placement_assessment_v1.py',
    'tools/train_curriculum_placement_course_v1.py','tools/evaluate_curriculum_placement_course_v1.py',
    'tools/plot_curriculum.py')


def extend(files,other):
    for path,expected in other.items():
        if path in files and files[path]!=expected:raise ValueError('Conflicting immutable input: '+path)
        files[path]=expected


def require_integration(proof,reg,job):
    if (proof['status']!='actual_placement_upstream_two_update_integration_passed'
        or proof['optimizer_steps']!=2 or proof['actual_control_record_exposures']!=114
        or proof['control_sources']!=57 or proof['complete_prepared_epoch_orders_exercised']!=2
        or str(proof['job_id'])!=str(job) or proof['node'].startswith(('lo-','login'))
        or proof['parent_identity']!=reg['parent_model']['identity_sha256']
        or any(proof[name] is not True for name in ('actual_updates_proven','parent_tensors_unchanged',
            'isolated_from_student_fit','actual_weighted_loss_journal_matches','saved_optimizer_states_verified',
            'fixtures_from_current_TRAIN_only','control_validation_not_original_VAL',
            'control_distribution_is_not_full_course_mix','control_weights_are_not_a_candidate'))
        or proof['fit_launch_admitted'] or proof['automatic_release']):
        raise ValueError('Actual isolated57-source114-exposure two-update integration required')
    exact=proof['exact_inheritance']
    if (not exact['parent_adapter_exact'] or not exact['parent_head_exact']
        or exact['adapter_tensors']!=372 or exact['head_tensors']!=4):
        raise ValueError('Actual exact qualified-parent tensor inheritance required')


def require_control_state(state,step):
    if (state['optimizer_step']!=step or state['records_seen']!=57*step
        or state['args'].get('resume') or len(state['optimizer']['state'])!=376
        or any(float(item['step'])!=step for item in state['optimizer']['state'].values())):
        raise ValueError('Durable new376-state control optimizer and exact exposure required')


def terminal(job,receipt,script,archive):
    """Preserve original scheduler output; never synthesize expired records."""
    if str(receipt.get('job_id'))!=str(job):raise ValueError('Terminal receipt belongs to another job')
    if archive.exists():
        result=read(archive)
        if result['receipt']!=receipt:raise ValueError('Archived terminal receipt differs')
    else:
        observed=subprocess.run(['scontrol','-o','show','job',str(job)],capture_output=True,text=True,timeout=20)
        if observed.returncode==0:
            scheduler_exit(receipt,observed.stdout,script)
            result=dict(source='actual_terminal_scontrol',raw=observed.stdout,receipt=receipt)
        else:
            actual=subprocess.run(['sacct','-X','-n','-P','-j',str(job),
                '--format=JobIDRaw,State,ExitCode,NodeList,AllocTRES,Start,End,SubmitLine'],
                capture_output=True,text=True,timeout=20)
            if actual.returncode!=0:raise RuntimeError('Real terminal record unavailable; no launch admission')
            accounting_proof(actual.stdout,receipt,job,script)
            result=dict(source='sacct_original_row',raw=actual.stdout,receipt=receipt)
        write(archive,result)
    if result['source']=='actual_terminal_scontrol':scheduler_exit(receipt,result['raw'],script)
    elif result['source']=='sacct_original_row':accounting_proof(result['raw'],receipt,job,script)
    else:raise ValueError('Unknown terminal evidence source')
    return result


def prepare(config):
    gpu_only();c=read(config)
    run_root = ROOT/'runs/curriculum-placement-paired-v1'
    queue_path = ROOT/'runs/curriculum-progression-riichi-paired-v1/placement-continuation-queue-private.json'
    queue = read(queue_path)
    if (c['status']!='planned_full_placement_launch_after_actual_integration'
        or c['integration_job']!=int(queue['integration_job'])
        or (ROOT/c['registered']).resolve()!=run_root/'fit-plan/registered.json'
        or (ROOT/c['integration_out']).resolve()!=run_root/'upstream-integration-v1'
        or c['automatic_release'] or c['open_independent_confirmation']):
        raise ValueError('Exact prospective dependent launch recipe required')
    registered=(ROOT/c['registered']).resolve();reg=read(registered);require_registration(reg)
    run=registered.parent.parent;fit=ROOT/reg['curriculum']['out'];launch=run/'launch-binding-private.json'
    if fit.exists() or launch.exists():raise ValueError('Never overwrite or silently resume a student fit')
    integration=(ROOT/c['integration_out']).resolve();proof_path=integration/'integration-proof-private.json'
    proof=read(proof_path);require_integration(proof,reg,c['integration_job'])
    binding_path=integration/'completion-binding-private.json';files=dict(reg['inputs'])
    extend(files,read(binding_path)['files'])
    for path in (registered,binding_path,proof_path):extend(files,{str(path):sha(path)})
    criteria_path=(ROOT/c['criteria_config']).resolve();require_criteria(read(criteria_path))
    for path in [config,criteria_path,queue_path,*[ROOT/name for name in PUBLIC_FILES]]:
        extend(files,{str(Path(path).resolve()):sha(path)})
    if any(sha(path)!=expected for path,expected in files.items()):raise ValueError('Bound integration/source changed')
    receipt_path=ROOT/'logs'/f"curriculum-placement-paired-exit-{c['integration_job']}.json"
    receipt=read(receipt_path)
    if (str(receipt['job_id'])!=str(c['integration_job']) or receipt['exit_code']!=0
        or receipt.get('phase')!='integration'):
        raise ValueError('Actual successful integration job receipt required')
    archive=run/'integration-terminal-scheduler-private.json'
    terminal(c['integration_job'],receipt,ROOT/'cluster/curriculum-placement-paired-v1.slurm',archive)
    for path in (receipt_path,archive):extend(files,{str(path.resolve()):sha(path)})
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from safetensors.torch import load_file
    from jevmahjong.evaluation.unified_contract import bundle
    from tools.curriculum_placement_metrics_v1 import summarize
    verify_parent(reg['training_config']);parent=bundle(reg['parent_model']['directory'])
    args=tomllib.loads((integration/'control.toml').read_text())
    if (args.get('resume') or Path(args['init_from']).resolve()!=Path(parent['directory']).resolve()
        or args['epochs']!=2 or args['max_steps']!=2 or args['batch']!=1 or args['accum']!=57
        or Path(args['out']).resolve()!=integration/'control-run'):
        raise ValueError('Actual new-optimizer isolated control route differs')
    metrics=read(integration/'control-run/training_metrics.json')
    if metrics['optimizer_steps']!=2 or metrics['records_seen']!=114 or metrics['rejected_records']:
        raise ValueError('Actual complete control training metrics required')
    for step in (1,2):
        state=torch.load(integration/'control-run/checkpoints'/f'step-{step:06d}'/'trainer_state.pt',
            map_location='cpu',weights_only=True)
        require_control_state(state,step)
    final=integration/'control-run/checkpoints/step-000002';parent_path=Path(parent['directory'])
    a=load_file(parent_path/'adapter_model.safetensors');b=load_file(final/'adapter_model.safetensors')
    ha=torch.load(parent_path/'head.pt',map_location='cpu',weights_only=True)['head']
    hb=torch.load(final/'head.pt',map_location='cpu',weights_only=True)['head']
    if (set(a)!=set(b) or set(ha)!=set(hb) or not any(not torch.equal(a[n],b[n]) for n in a)
        or not any(not torch.equal(ha[n],hb[n]) for n in ha)):
        raise ValueError('Actual isolated LoRA and head updates missing')
    del a,b,ha,hb,state
    journal=read(integration/'weighted-loss-journal-private.json')
    if [row['step'] for row in journal]!=[1,2]:raise ValueError('Both actual control objective observations required')
    for row in journal:
        pending=[dict(source=source,**values) for source,values in row['source_loss_components'].items()]
        actual=summarize(pending,reg['course_loss_weights'])
        if (len(pending)!=57 or set(row['source_counts'].values())!={1}
            or abs(actual['loss']-row['loss'])>1e-5
            or abs(row['raw_upstream_metrics']['train/objective']-row['loss'])>1e-5):
            raise ValueError('Actual control loss decomposition/upstream objective differs')
    orders=[read(integration/'full-order-control-private'/f'epoch-{i:02d}.json') for i in (1,2)]
    if orders!=reg['expected_orders']:raise ValueError('Actual full production order adapter was not exercised')
    sealed=read(reg['curriculum']['parent_seal']);parent_reg=read(sealed['parent_registration'])
    parent_args=tomllib.loads(Path(parent_reg['training_config']['trainer_config']).read_text())
    validation=[json.loads(line) for line in Path(parent_args['eval_data']).open()]
    saved=read(ROOT/parent_reg['curriculum']['out']/'training-run/training_eval/step-002500/development/rows.json')
    coverage=early_profile(validation,saved)
    if any(v is None for key in ('plain_agreement','RI_balanced_accuracy','original_call_balanced_accuracy') for v in coverage[key].values()):
        raise ValueError('Original early positive/negative retention coverage incomplete')
    if any(sha(path)!=expected for path,expected in files.items()) or bundle(parent_path)!=parent:
        raise ValueError('Frozen source/data/parent changed during launch preparation')
    result=dict(status='placement_actual_upstream_integration_and_launch_bound_v1',files=files,
        registered_sha256=sha(registered),fit_launch_admitted=True,integration_proof=str(proof_path),
        integration_job=c['integration_job'],criteria_config=str(criteria_path),
        parent_identity=parent['identity_sha256'],optimizer_updates_at_binding=0,
        candidate_rule='registered_full_two_epoch_final_only',
        parent_game_cache_as_child_zero_allowed=False,early_retention_scope=coverage,
        final_reserved_grid_opened=False,automatic_release=False,
        node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'])
    write(launch,result);print(json.dumps(dict(status=result['status'],fit_launch_admitted=True,
        registered_final_step=reg['registered_final_step'],new_student_optimizer_updates=0)),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    prepare(p.parse_args().config.resolve())
