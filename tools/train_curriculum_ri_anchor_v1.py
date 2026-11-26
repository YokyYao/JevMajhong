"""Registered augmented RI retention; actual full-budget training and curves."""
import argparse
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_ri_anchor_runtime_v1 import verify_parent,parent_loader,require_registration,policy_contract,prediction_parity,require_launch,profile,CACHE,POINTS
from tools.curriculum_ri_balanced_replay_runtime_v1 import expand_after_unique_admission
from tools.curriculum_progression_parent_loss_v1 import annotate
from tools.curriculum_ri_parent_anchor_weighted_v1 import METHOD,weighted_loss,summarize
from tools.curriculum_ri_replay_epoch_adapter_v1 import random_proxy
from tools.curriculum_ri_parent_anchor_epoch_v2 import AnchoredEpochReplay
from tools.curriculum_ri_parent_anchor_v1 import make_targets
from tools.curriculum_ri_canonical_replay_v2 import verify_metadata_only
from tools.curriculum_checkpoint_evaluation_fence_v1 import CheckpointEvaluationFence, checkpoint_ready


def train(registered):
    gpu_only(); reg = read(registered); require_registration(reg); require_launch(registered); c = reg['curriculum']
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevany import train as trainer
    from jevmahjong.training import base_lora, initialization
    from jevmahjong.training.metric_journal import MetricJournal, LocalTracker, export_csv
    from jevmahjong.evaluation.unified_contract import bundle
    launch = read(registered.parent.parent/'launch-binding-private.json')
    if launch['registered_sha256'] != sha(registered): raise ValueError('Actual launch registration differs')
    for p, h in launch['files'].items():
        if sha(p) != h: raise ValueError('Frozen launcher changed: '+p)
    def verify():
        for p, h in reg['inputs'].items():
            if sha(p) != h: raise ValueError('Frozen training input changed: '+p)
    verify(); verify_parent(reg['training_config'])
    if base_lora.package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']:
        raise ValueError('Pinned runtime changed')
    out = ROOT/c['out']; metrics = out/'metrics'
    if out.exists(): raise ValueError('No silent restart or overwrite of a fit')
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    rows = [json.loads(line) for line in Path(args['data']).open()]
    plans = [read(p) for p in reg['replay_epoch_plans']]
    canonical=trainer.load_records(args['data']);canonical_proof=verify_metadata_only(rows,canonical)
    targets=make_targets(rows,[json.loads(line) for line in (CACHE/'parent-TRAIN-predictions-private.jsonl').open()])
    replay=AnchoredEpochReplay(canonical,plans,targets)
    original = dict(random=trainer.random, batch=trainer.batch_loss, tracker=trainer.start_wandb,
        requests=trainer.training_requests, materialize=trainer.materialize, encode=trainer.encode_batch,
        loss=base_lora.question_loss, profile=base_lora.validation_profile, evaluate=trainer.evaluate_during_training)
    trainer.random, rngs = random_proxy(original['random'], c['seed'], replay)
    trainer.materialize = lambda request: annotate(original['materialize'](request), request)
    trainer.encode_batch = lambda model, tok, a, chunk, epoch: replay.encode_batch(original['encode'], model, tok, a, chunk, epoch)
    pending = []; queued = set(); finished = set(); contracts = {}; warm = None

    def game_eval(step):
        nonlocal warm
        checkpoint = out/'training-run/checkpoints'/f'step-{step:06d}'
        if not (checkpoint/'head.pt').is_file(): raise ValueError('Actual saved checkpoint required')
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
            parent_predictions = read(ROOT/parent['curriculum']['out']/'training-run/training_eval/step-000388/development/rows.json')
            current_predictions = read(out/'training-run/training_eval/step-000000/development/rows.json')
            parity = prediction_parity(parent_rows, parent_predictions, child_rows, current_predictions)
            warm = dict(exact_tensors=exact, all_parent_prediction_parity=parity,
                parent_bundle_identity=c['parent_identity'], parent_policy_identity=c['parent_policy_identity'],
                step0_bundle_identity=identity['identity_sha256'], step0_policy_identity=policy_id,
                checked_before_first_optimizer_update=True)
            write(out/'warmstart-validation-private.json', warm)
        for label, cache in (('reference-parent', c['parent_evaluation_cache']),
                             ('reference-original-guarded', c['original_guarded_evaluation_cache']),
                             ('reference-v8', c['reference_evaluation_cache'])):
            target = out/'evaluations'/label; source = (ROOT/cache).resolve()
            expected = c['reference_identity'] if label == 'reference-v8' else c['parent_identity']
            if read(source/'summary.json')['checkpoint_sha256'] != expected: raise ValueError('Reference cache parameter identity differs')
            if label == 'reference-parent' and read(source/'summary.json')['policy_identity'] != c['parent_policy_identity']:
                raise ValueError('Actual changed-policy parent cache required')
            if not target.exists(): target.parent.mkdir(parents=True, exist_ok=True); target.symlink_to(source, target_is_directory=True)
        if step==0:
            source=ROOT/c['step0_native_reuse_cache']; directory=out/'evaluations/step-000000'
            directory.mkdir(parents=True,exist_ok=False)
            shutil.copyfile(source/'matches-private.jsonl',directory/'matches-private.jsonl')
            protocol=dict(read(source/'protocol.json'),checkpoint=identity,policy=policy,policy_identity=policy_id,
                actual_numeric_identity_cache_reuse=True,source_cache=str(source))
            summary=dict(read(source/'summary.json'),checkpoint_sha256=identity['identity_sha256'],
                policy_identity=policy_id,stage=c['stage'],
                source='actual_prior_step0_games_reused_after_exact_numeric_identity_proof')
            write(directory/'protocol.json',protocol);write(directory/'summary.json',summary)
            MetricJournal(metrics,c['run_id'],c['stage'],c['stage_offset']).append('evaluation',0,
                **{k:v for k,v in summary.items() if k!='stage'})
        else:
            subprocess.run([sys.executable, str(ROOT/'tools/evaluate_curriculum_ri_anchor_v1.py'), '--registered', str(registered),
                '--checkpoint', str(checkpoint), '--step', str(step)], check=True, timeout=1800)
        if step == 0:
            reference = [json.loads(line) for line in (ROOT/c['parent_evaluation_cache']/'matches-private.jsonl').open()]
            expected = [r for r in reference if r['seed'] < c['curve_seed_start']+2]
            actual = [json.loads(line) for line in (out/'evaluations/step-000000/matches-private.jsonl').open()]
            if len(actual) != 16 or actual != expected: raise ValueError('Complete-legal parent16 full event/outcome reproduction failed')
            write(out/'warmstart-games-private.json', dict(matches=16, exact_complete_legal_parent_events=True,
                checked_before_first_update=True, original_guarded_policy_not_equated=True, parent_policy_identity=c['parent_policy_identity']))
        finished.add(step)

    fence=CheckpointEvaluationFence(lambda step:checkpoint_ready(out/'training-run/checkpoints',step),game_eval)
    queued=fence.queued;finished=fence.finished
    def flush():fence.flush()
    def batch(*a, **kw):
        flush()
        if warm is None or 0 not in finished: raise ValueError('All real warm-start proofs must precede updates')
        return original['batch'](*a, **kw)
    def admitted(*a, **kw):
        result = original['requests'](*a, **kw)
        if [r['_meta']['id'] for r in result[0]] != [r['_meta']['id'] for r in rows]:
            raise ValueError('Actual training token admission changed original row order/membership')
        pool, unique_report, expanded = expand_after_unique_admission(replay, *result)
        # The pinned base trainer checks admission of unique raw inputs. Keep
        # that real report; record virtual physical slots separately, explicitly.
        write(out/'physical-exposure-admission-private.json', expanded)
        write(out/'canonical-request-admission-private.json',dict(canonical_proof,
            actual_unique_token_admission=result[1],strict_full_canonical_object_equality_passed=True,
            parent_targets_attached_after_actual_encoding=True))
        return pool, unique_report
    def validation(records, predictions, ignored):
        return profile(records, predictions, {}, base_lora.target_cross_entropy)

    class Tracker(LocalTracker):
        def log(self, values):
            step = values['optimizer_step']
            if 'train/objective' in values:
                actual = summarize(pending)
                if abs(actual['loss']-values['train/objective']) > 1e-5: raise ValueError('Actual optimizer objective differs from recorded components')
                self.journal.append('train', step, **actual, reward=None, lr=values['train/lr'],
                    lr_position='post_scheduler_next_update', grad_norm=values['train/grad_norm_pre_clip'],
                    head_lr_derived=values['train/lr']*c['head_learning_rate']/c['learning_rate'],
                    records_seen=values['train/records_seen'], raw_upstream_metrics=values)
                pending.clear()
            else:
                point = dict(read(out/'validation-curve.json')[-1])
                if point.pop('step') != step: raise ValueError('Actual validation/journal step differs')
                self.journal.append('validation', step, loss=point['weighted_class_objective'], ce=point['weighted_target_ce'],
                    validation_data_sha256=read(c['data_ready'])['files']['validation']['sha256'], **point)
                if step in c['validation_and_game_steps']:fence.validation(step)
    trainer.batch_loss = batch; trainer.training_requests = admitted
    trainer.start_wandb = lambda a, d: Tracker(MetricJournal(metrics, c['run_id'], c['stage'], c['stage_offset']), c['run_id'])
    base_lora.question_loss = lambda z, q, device: weighted_loss(z, q, device, pending)
    base_lora.validation_profile = validation
    try:
        with parent_loader(initialization): base_lora.train(reg['training_config'])
        flush(); actual = read(out/'training-run/training_metrics.json')
        if (actual['optimizer_steps'] != 2596 or actual['records_seen'] != 189508 or pending
                or queued != finished or finished != set(POINTS) or len(rngs) != 1):
            raise ValueError('Incomplete actual two-epoch exposure/curves')
        write(out/'actual-replay-mapping-private.json', replay.finish())
        verify(); export_csv(metrics)
        write(out/'policy-contracts-private.json', {str(k): v for k, v in contracts.items()})
        for name in ('plot_curriculum.py', 'plot_curriculum_fixed_grid.py'):
            subprocess.run([sys.executable, str(ROOT/'tools'/name), '--metrics-dir', str(metrics)], check=True, timeout=180)
        write(out/'completion.json', dict(status='ri_anchor2596_fit_and_real_curves_complete', optimizer_steps=2596,
            actual_records_seen=189508, unique_train_rows=92570, curve_steps=sorted(finished), loss_method=METHOD,
            parent_identity=c['parent_identity'], parent_policy_identity=c['parent_policy_identity'],
            complete_legal_execution_transfer_explicit=True, automatic_stage4_started=False, automatic_release=False))
    finally:
        trainer.random = original['random']; trainer.batch_loss = original['batch']; trainer.start_wandb = original['tracker']
        trainer.training_requests = original['requests']; trainer.materialize = original['materialize']
        trainer.encode_batch = original['encode']
        base_lora.question_loss = original['loss']; base_lora.validation_profile = original['profile']
        trainer.evaluate_during_training=original['evaluate']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registered', type=Path, required=True)
    a = p.parse_args(); train(a.registered.resolve())

