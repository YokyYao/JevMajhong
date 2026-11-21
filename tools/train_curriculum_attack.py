"""Continue the qualified calls adapter/head with masked attack/risk targets."""
from collections import defaultdict
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_parent import read, sha
from tools.curriculum_attack_losses import PREFIX, canonical_question
from tools.curriculum_attack_profile import profile as attack_profile
from tools.curriculum_calls_parent import calls_parent_loader
from tools.curriculum_losses import question_loss


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def run(registered):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-', 'login'))
    reg = read(registered)
    c = reg['curriculum']
    if c.get('riichi_input_schema') is not None:
        raise ValueError('This registered attack route uses the qualified plain-input calls parent')
    sys.path.insert(0, reg['source'])
    sys.path.insert(0, reg['training_config']['upstream_source'])
    os.chdir(ROOT)
    import torch
    torch.set_num_threads(4)
    from jevany import train as trainer
    from jevmahjong.training import base_lora, initialization
    from jevmahjong.training.metric_journal import MetricJournal, LocalTracker, export_csv
    from tools.curriculum_course_order import trainer_course_random_proxy
    if base_lora.package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']:
        raise ValueError('Pinned source changed')
    def verify():
        for p, h in reg['inputs'].items():
            if sha(p) != h:
                raise ValueError('Registered attack input changed: '+p)
    verify()
    out = ROOT/c['out']
    metrics = out/'metrics'
    last = reg['registered_final_step']
    if out.exists():
        raise ValueError('Existing attack fit must never be silently restarted')
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    rows = [json.loads(s) for s in Path(args['data']).read_text().splitlines() if s]
    categories = [r['questions']['action']['src'].removeprefix(PREFIX) for r in rows]
    originals = dict(random=trainer.random, batch=trainer.batch_loss, tracker=trainer.start_wandb,
        requests=trainer.training_requests, loss=base_lora.question_loss, profile=base_lora.validation_profile)
    trainer.random, epoch_rng = trainer_course_random_proxy(originals['random'], c['seed'], categories, c['accumulation'], out/'epoch-order-private')
    pending, queue, finished = [], set(), set()
    def game_eval(step):
        checkpoint = out/'training-run/checkpoints'/f'step-{step:06d}'
        if not checkpoint.exists():
            return
        for label, source, expected in [('reference-v8', c['reference_evaluation_cache'], c['reference_identity']),
                                         ('reference-parent', c['parent_evaluation_cache'], c['parent_identity'])]:
            target = out/'evaluations'/label
            source = (ROOT/source).resolve()
            summary = read(source/'summary.json')
            protocol = read(source/'protocol.json')
            if (summary['checkpoint_sha256'] != expected or summary['matches'] != 64
                or protocol['seeds'] != list(range(c['curve_seed_start'], c['curve_seed_start']+c['stage_end_seed_groups']))
                or protocol['profiles'] != c['opponent_profiles']):
                raise ValueError('Attack reference cache grid differs')
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source, target_is_directory=True)
        subprocess.run([sys.executable, str(ROOT/'tools/evaluate_curriculum.py'), '--registered', str(registered),
            '--checkpoint', str(checkpoint), '--step', str(step), '--groups',
            str(c['stage_end_seed_groups'] if step == last else c['curve_seed_groups'])], check=True, timeout=1200)
        if step == 0:
            actual = read(out/'evaluations/step-000000/summary.json')
            expected = read(out/'evaluations/reference-parent/summary.json')['curve_grid_subset']
            if any(actual.get(k) != v for k, v in expected.items()):
                raise ValueError('Step0 native games do not reproduce the same parent parameter policy')
        finished.add(step)
    def flush():
        for step in sorted(queue-finished):
            game_eval(step)
    def batch(*a, **kw):
        flush()
        return originals['batch'](*a, **kw)
    def admitted(*a, **kw):
        result = originals['requests'](*a, **kw)
        if [r['_meta']['id'] for r in result[0]] != [r['_meta']['id'] for r in rows]:
            raise ValueError('Token admission changed the registered attack rows/order')
        return result
    def loss(logits, question, device):
        kind, canonical = canonical_question(question)
        if kind not in reg['course_loss_weights']:
            raise ValueError('Unregistered attack loss stratum')
        ce = question_loss(logits, canonical, device)
        if torch.is_grad_enabled() and logits.requires_grad:
            pending.append((kind, float(ce.detach())))
        return ce*reg['course_loss_weights'][kind]
    def profile(records, predictions, weights):
        return attack_profile(records, predictions, weights, reg['course_loss_mass'])
    class Tracker(LocalTracker):
        def log(self, values):
            step = values['optimizer_step']
            if 'train/objective' in values:
                by = defaultdict(list)
                for kind, ce in pending:
                    by[kind].append(ce)
                if set(by) != set(reg['course_loss_mass']):
                    raise ValueError('An actual attack update lacks a registered stratum')
                self.journal.append('train', step, loss=values['train/objective'], ce=values['train/ce'], reward=None,
                    lr=values['train/lr'], lr_position='post_scheduler_next_update',
                    head_lr_derived=values['train/lr']*c['head_learning_rate']/c['learning_rate'],
                    grad_norm=values['train/grad_norm_pre_clip'], records_seen=values['train/records_seen'],
                    source_ce={k: sum(v)/len(v) for k, v in by.items()}, source_counts={k: len(v) for k, v in by.items()},
                    raw_upstream_metrics=values)
                pending.clear()
            else:
                point = read(out/'validation-curve.json')[-1]
                if point['step'] != step:
                    raise ValueError('Attack validation event step differs')
                fields = {k: v for k, v in point.items() if k not in ('step', 'upstream_hard_label_nll')}
                self.journal.append('validation', step, loss=point['weighted_target_ce'], **fields,
                    validation_data_sha256=read(reg['training_config']['data_receipt'])['files']['validation']['sha256'])
                if step == 0 or step % c['game_evaluation_every_steps'] == 0 or step == last:
                    queue.add(step)
                flush()
    trainer.batch_loss = batch
    trainer.training_requests = admitted
    trainer.start_wandb = lambda a, d: Tracker(MetricJournal(metrics, c['run_id'], c['stage'], c['stage_offset']), c['run_id'])
    base_lora.question_loss = loss
    base_lora.validation_profile = profile
    try:
        with calls_parent_loader(initialization):
            base_lora.train(reg['training_config'])
        flush()
        actual = read(out/'training-run/training_metrics.json')
        if actual['optimizer_steps'] != last or actual['records_seen'] != reg['train_records']*c['epochs']:
            raise ValueError('Attack fit did not finish the full registered exposure')
        if queue != finished or len(epoch_rng) != 1 or epoch_rng[0].epochs != c['epochs']:
            raise ValueError('Missing actual epoch/game evidence')
        verify()
        export_csv(metrics)
        for name in ('plot_curriculum.py', 'plot_curriculum_fixed_grid.py'):
            subprocess.run([sys.executable, str(ROOT/'tools'/name), '--metrics-dir', str(metrics)], check=True, timeout=180)
        write(out/'completion.json', dict(status='stage3_fit_and_curves_complete', optimizer_steps=last,
            actual_records_seen=actual['records_seen'], curve_steps=sorted(finished), new_optimizer=True,
            parent_identity=c['parent_identity'], candidate_rule=c['candidate_rule'],
            historical_minimum_CE_result_not_course_candidate=True, automatic_stage4_started=False, automatic_release=False))
    finally:
        trainer.random = originals['random']
        trainer.batch_loss = originals['batch']
        trainer.start_wandb = originals['tracker']
        trainer.training_requests = originals['requests']
        base_lora.question_loss = originals['loss']
        base_lora.validation_profile = originals['profile']


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered', type=Path, required=True)
    run(p.parse_args().registered.resolve())
