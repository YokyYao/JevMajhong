"""Full two-epoch placement-course fit; requires a separately bound launch.

The immutable budget registration alone never admits optimizer updates.
Production/released checkpoints are neither overwritten nor selected here.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_placement_retained_parent_v1 import verify_parent, parent_loader
from tools.curriculum_placement_parent_runtime_v1 import require_registration, policy_contract, prediction_parity, random_proxy
from tools.curriculum_placement_parent_loss_v2 import METHOD, annotate, weighted_loss
from tools.curriculum_placement_parent_metrics_v2 import summarize
from tools.curriculum_placement_parent_metrics_v2 import partition_profile as profile
from tools.curriculum_checkpoint_evaluation_fence_v1 import CheckpointEvaluationFence, checkpoint_ready


def train(registered):
    gpu_only(); reg = read(registered); require_registration(reg); c = reg['curriculum']
    launch = read(registered.parent.parent/'launch-binding-private.json')
    if (launch['status'] != 'placement_retained_objective_actual_upstream_integration_and_launch_bound_v1'
            or launch['registered_sha256'] != sha(registered) or not launch['fit_launch_admitted']
            or launch['automatic_release'] or launch['final_reserved_grid_opened']):
        raise ValueError('Actual isolated upstream integration and complete launch binding required')
    for path, expected in launch['files'].items():
        if sha(path) != expected: raise ValueError('Frozen launch input changed: '+path)
    proof = read(launch['integration_proof'])
    if (proof['status'] != 'actual_placement_upstream_two_update_integration_passed'
            or proof['optimizer_steps'] != 2 or not proof['actual_updates_proven']
            or not proof['parent_tensors_unchanged'] or not proof['isolated_from_student_fit']
            or not proof['actual_weighted_loss_journal_matches']
            or not proof['saved_optimizer_states_verified'] or proof['loss_method'] != METHOD):
        raise ValueError('Real upstream updates/optimizer/journal integration must pass before full fit')
    if str(Path(launch['integration_proof']).resolve()) not in launch['files']:
        raise ValueError('Integration evidence itself must be immutable in the launch binding')
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevany import train as trainer
    from jevmahjong.training import base_lora, initialization
    from jevmahjong.training.metric_journal import MetricJournal, LocalTracker, export_csv
    from jevmahjong.evaluation.unified_contract import bundle
    def verify():
        for path, expected in reg['inputs'].items():
            if sha(path) != expected: raise ValueError('Frozen course input changed: '+path)
        for path, expected in launch['files'].items():
            if sha(path) != expected: raise ValueError('Frozen launch input changed: '+path)
    verify(); verify_parent(reg['training_config'])
    if base_lora.package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']:
        raise ValueError('Pinned training runtime changed')
    out = ROOT/c['out']; metrics = out/'metrics'
    if out.exists(): raise ValueError('No silent fit restart or overwrite')
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    rows = [json.loads(line) for line in Path(args['data']).open()]
    if len(rows) != reg['train_records']:
        raise ValueError('Actual complete expanded TRAIN corpus required')
    categories = [r['questions']['action']['src'] for r in rows]
    membership = read(read(c['parent_seal'])['child_membership_file'])
    original = dict(random=trainer.random, batch=trainer.batch_loss, tracker=trainer.start_wandb,
        requests=trainer.training_requests, materialize=trainer.materialize,
        loss=base_lora.question_loss, profile=base_lora.validation_profile)
    trainer.random, rngs = random_proxy(original['random'], c['seed'], categories, c['accumulation'],
        out/'epoch-order-private', reg['expected_orders'])
    trainer.materialize = lambda request: annotate(original['materialize'](request), request)
    pending = []; contracts = {}; warm = None

    def game_eval(step):
        nonlocal warm
        checkpoint = out/'training-run/checkpoints'/f'step-{step:06d}'
        if not checkpoint_ready(out/'training-run/checkpoints', step):
            raise ValueError('Complete saved checkpoint required before native evaluation')
        identity = bundle(checkpoint); policy, policy_id = policy_contract(identity, reg)
        contracts[step] = dict(model_identity=identity['identity_sha256'], policy=policy, policy_identity=policy_id)
        if step == 0 and warm is None:
            from safetensors.torch import load_file
            exact = initialization.exact_loaded_tensors(load_file(checkpoint/'adapter_model.safetensors'),
                torch.load(checkpoint/'head.pt', map_location='cpu', weights_only=True)['head'], reg['parent_model']['directory'])
            sealed = read(c['parent_seal']); parent = read(sealed['parent_registration'])
            pa = tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())
            parent_rows = [json.loads(line) for line in Path(pa['eval_data']).open()]
            child_rows = [json.loads(line) for line in Path(args['eval_data']).open()]
            parent_predictions = read(ROOT/parent['curriculum']['out']/'training-run/training_eval/step-002500/development/rows.json')
            current_predictions = read(out/'training-run/training_eval/step-000000/development/rows.json')
            parity = prediction_parity(parent_rows, parent_predictions, child_rows, current_predictions)
            warm = dict(exact_tensors=exact, early_prediction_parity=parity,
                parent_bundle_identity=c['parent_identity'], parent_policy_identity=c['parent_policy_identity'],
                step0_bundle_identity=identity['identity_sha256'], step0_policy_identity=policy_id,
                checked_before_first_optimizer_update=True)
            write(out/'warmstart-validation-private.json', warm)
        subprocess.run([sys.executable, str(ROOT/'tools/evaluate_curriculum_placement_parent_v1.py'),
            '--registered', str(registered), '--checkpoint', str(checkpoint), '--step', str(step)],
            check=True, timeout=2400)
        if step == 0:
            summary = read(out/'evaluations/step-000000/summary.json')
            if (summary['matches'] != 64 or summary['actual_new_policy_matches'] != 64
                    or summary['parent_cache_reused_as_step0'] or summary['policy_identity'] != policy_id):
                raise ValueError('Fresh64-game child-policy baseline required before updating')
            write(out/'warmstart-games-private.json', dict(matches=64, actual_new_child_policy_games=True,
                parent_game_cache_reused=False, checked_before_first_update=True, policy_identity=policy_id))

    fence = CheckpointEvaluationFence(lambda step: checkpoint_ready(out/'training-run/checkpoints', step), game_eval)
    def batch(*a, **kw):
        fence.flush()
        if warm is None or 0 not in fence.finished:
            raise ValueError('Actual inheritance and fresh child-policy games must precede updates')
        return original['batch'](*a, **kw)
    def admitted(*a, **kw):
        result = original['requests'](*a, **kw)
        if [r['_meta']['id'] for r in result[0]] != [r['_meta']['id'] for r in rows]:
            raise ValueError('Actual token admission changed expanded membership or order')
        return result
    def validation(records, predictions, ignored):
        return profile(records, predictions, {}, base_lora.target_cross_entropy, membership)

    class Tracker(LocalTracker):
        def log(self, values):
            step = values['optimizer_step']
            if 'train/objective' in values:
                actual = summarize(pending, reg['course_loss_weights'])
                if abs(actual['loss']-values['train/objective']) > 1e-5:
                    raise ValueError('Actual optimizer objective differs from recorded components')
                self.journal.append('train', step, **actual, reward=None, lr=values['train/lr'],
                    lr_position='post_scheduler_next_update', grad_norm=values['train/grad_norm_pre_clip'],
                    head_lr_derived=values['train/lr']*c['head_learning_rate']/c['learning_rate'],
                    records_seen=values['train/records_seen'], raw_upstream_metrics=values)
                pending.clear()
            else:
                point = dict(read(out/'validation-curve.json')[-1])
                if point.pop('step') != step: raise ValueError('Actual validation/journal steps differ')
                self.journal.append('validation', step, loss=point['weighted_class_objective'],
                    ce=point['weighted_target_ce'],
                    validation_data_sha256=read(c['data_ready'])['files']['validation']['sha256'], **point)
                if step in c['validation_and_game_steps']: fence.validation(step)
    trainer.batch_loss = batch; trainer.training_requests = admitted
    trainer.start_wandb = lambda a, d: Tracker(MetricJournal(metrics, c['run_id'], c['stage'], c['stage_offset']), c['run_id'])
    base_lora.question_loss = lambda z, q, device: weighted_loss(z, q, device, reg['course_loss_weights'], pending)
    base_lora.validation_profile = validation
    try:
        with parent_loader(initialization): base_lora.train(reg['training_config'])
        fence.flush(); actual = read(out/'training-run/training_metrics.json')
        if (actual['optimizer_steps'] != reg['registered_final_step']
                or actual['records_seen'] != reg['expected_record_exposures'] or pending
                or fence.queued != fence.finished or fence.finished != set(c['validation_and_game_steps'])
                or len(rngs) != 1 or rngs[0].epochs != 2):
            raise ValueError('Incomplete actual two-epoch exposure budget or real curves')
        verify(); export_csv(metrics)
        write(out/'policy-contracts-private.json', {str(k): v for k, v in contracts.items()})
        subprocess.run([sys.executable, str(ROOT/'tools/plot_curriculum.py'), '--metrics-dir', str(metrics)],
            check=True, timeout=180)
        write(out/'completion.json', dict(status='placement_full_budget_fit_and_real_curves_complete',
            optimizer_steps=actual['optimizer_steps'], actual_records_seen=actual['records_seen'],
            curve_steps=sorted(fence.finished), loss_method=METHOD, parent_identity=c['parent_identity'],
            parent_policy_identity=c['parent_policy_identity'], same_execution_policy_as_parent=False,
            new_child_policy_baseline64=True, curve_grid_matches=16,
            independent_confirmation_opened=False, automatic_release=False))
    finally:
        trainer.random = original['random']; trainer.batch_loss = original['batch']; trainer.start_wandb = original['tracker']
        trainer.training_requests = original['requests']; trainer.materialize = original['materialize']
        base_lora.question_loss = original['loss']; base_lora.validation_profile = original['profile']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registered', type=Path, required=True)
    a = p.parse_args(); train(a.registered.resolve())

