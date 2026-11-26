"""Prepare a separate frozen replay recipe; actual launch requires final failure."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha
from tools.curriculum_ri_balanced_replay_runtime_v1 import (
    CONTROL, RUN, METHOD, VALIDATION_METHOD, curriculum_view, trainer_view,
    require_registration, failed_control, require_launch)

PUBLIC = (
    'tools/curriculum_ri_balanced_replay_runtime_v1.py',
    'tools/register_curriculum_ri_balanced_replay_v1.py',
    'tools/train_curriculum_ri_balanced_replay_v1.py',
    'tools/evaluate_curriculum_ri_balanced_replay_v1.py',
    'tools/audit_curriculum_ri_balanced_replay_v1.py',
    'tools/launch_curriculum_ri_balanced_replay_v1.py',
    'cluster/curriculum-ri-balanced-replay-v1.slurm',
    'tests/core/test_curriculum_ri_balanced_replay_v1.py',
    'docs/experiments/plans/curriculum-ri-balanced-replay-fit-20261125.md',
)


def register():
    allocation = resource()
    if RUN.exists(): raise ValueError('Never overwrite a prospective registration')
    source = CONTROL/'fit-plan/registered.json'; old = read(source)
    from tools.curriculum_progression_parent_control_v1 import verify
    verify(old)
    files = dict(old['inputs']); files[str(source)] = sha(source)
    bindings = [ROOT/'runs'/name/'completion-binding-private.json' for name in (
        'curriculum-ri-replay-epoch-adapter-v1', 'curriculum-ri-replay-model-probe-v1')]
    for binding in bindings:
        proof = read(binding)
        for p, h in proof['files'].items():
            if sha(p) != h: raise ValueError('Actual replay prerequisite changed: '+p)
            if p in files and files[p] != h: raise ValueError('Conflicting prerequisites')
            files[p] = h
        files[str(binding)] = sha(binding)
    result = subprocess.run([sys.executable, '-m', 'unittest',
        'tests.core.test_curriculum_ri_balanced_replay_v1'], capture_output=True, text=True, timeout=90)
    if result.returncode: raise RuntimeError(result.stdout+result.stderr)
    subprocess.run(['bash','-n',str(ROOT/'cluster/curriculum-ri-balanced-replay-v1.slurm')],check=True,timeout=15)
    import ast
    for name in PUBLIC:
        if name.endswith('.py'): ast.parse((ROOT/name).read_text(),filename=name)
    plan = RUN/'fit-plan'; plan.mkdir(parents=True)
    reg = copy.deepcopy(old); c = curriculum_view(old['curriculum'])
    args = trainer_view(tomllib.loads(Path(old['training_config']['trainer_config']).read_text()))
    trainer = plan/'trainer.toml'
    trainer.write_text(''.join(f'{k} = {json.dumps(v)}\n' for k, v in args.items()))
    cfg = dict(old['training_config'], model_id=c['run_id'], out=c['out'],
        trainer_config=str(trainer), trainer_config_sha256=sha(trainer),
        registered_final_step=2472, loss_method=METHOD)
    write(plan/'training-config.json', cfg)
    for p in [ROOT/p for p in PUBLIC]+[trainer, plan/'training-config.json']:
        files[str(p)] = sha(p)
    for key in ('expected_orders', 'parent_loss_control_registration', 'parent_loss_control_failed_grade',
                'only_replay_objective_changed_from_paired_control'):
        reg.pop(key, None)
    reg.update(status='balanced_replay2472_registered_not_started', curriculum=c, training_config=cfg,
        registered_final_step=2472, expected_record_exposures=180456,
        physical_train_slots_per_epoch=90228, sampling_control_registration=str(source),
        validation_loss_method=VALIDATION_METHOD, standby_only_after_control_failed_strategy=True,
        replay_epoch_plans=[str(ROOT/'runs/curriculum-ri-every-update-replay-v1'/f'epoch-{e:02d}-private.json') for e in (1,2)],
        inputs=files)
    require_registration(reg)
    registered = plan/'registered.json'; write(registered, reg)
    write(RUN/'launch-binding-private.json', dict(files={**files, str(registered): sha(registered)},
        registered_sha256=sha(registered), selected_initializer='qualified_calls388',
        no_current_3A_updates_in_initializer=True, new_optimizer_updates=0))
    write(RUN/'registration-evidence-private.json', dict(
        status='balanced_replay_recipe_registered_standby_not_fit_started', allocation=allocation,
        unit_test_returncode=result.returncode, unit_test_output=result.stdout+result.stderr,
        unique_train_rows=87595, physical_exposures=180456, optimizer_steps_planned=2472,
        parent_selected_step=388, student_updates=0, new_games=0,
        live_trial_changed=False, final_control_gate_not_yet_satisfied=True))
    print(json.dumps({'status':'balanced_replay_recipe_registered_standby_not_fit_started','student_updates':0}), flush=True)


def authorize():
    allocation = resource(); registered = RUN/'fit-plan/registered.json'
    require_registration(read(registered))
    gate_path = RUN/'launch-gate-private.json'
    if gate_path.exists(): raise ValueError('Never replace an actual launch decision')
    paths = dict(completion=str(CONTROL/'stage3-fit/completion.json'),
        audit=str(CONTROL/'final-evidence-private.json'), execution=str(CONTROL/'audit-execution-private.json'),
        capture=str(CONTROL/'parent-retention-terminal-v1/terminal-evidence-private.json'))
    grade = failed_control(*(read(paths[k]) for k in ('completion','audit','execution','capture')))
    capture_binding = CONTROL/'parent-retention-terminal-v1/completion-binding-private.json'
    files = dict(read(capture_binding)['files'])
    files[str(capture_binding)] = sha(capture_binding)
    files.update({p:sha(p) for p in paths.values()})
    for p, h in files.items():
        if sha(p) != h: raise ValueError('Actual terminal capture changed: '+p)
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    original_queue = read(ROOT/'runs/curriculum-progression-riichi-paired-v1/parent-retention-queue-v1-private.json')
    for role, proof in read(paths['capture'])['scheduler_proofs'].items():
        if proof['job_id'] != str(original_queue[role+'_job']):
            raise ValueError('Wrong actual original control scheduler job')
        scheduler_exit(proof['exit_receipt'], proof['raw'], ROOT/'cluster/curriculum-progression-parent-retention-v1.slurm')
    write(gate_path, dict(status='actual_failed_control_allows_one_separate_replay_fit',
        registered_sha256=sha(registered), evidence_paths=paths, files=files,
        unaltered_failed_control_grade=grade, allocation=allocation,
        initializer='qualified_calls388', optimizer_resumed=False, student_updates=0))
    require_launch(registered)
    print(json.dumps({'status':'actual_failed_control_allows_one_separate_replay_fit'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--phase', choices=('register','authorize'), required=True)
    a = p.parse_args()
    if a.phase == 'register': register()
    else: authorize()
