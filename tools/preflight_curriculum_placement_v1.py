"""Isolated two-update real-model integration, never a placement candidate.

Use TRAIN-only control fixtures across57 sources. The control's distribution
and accumulation differ from the production course; it verifies plumbing,
inheritance and durable updates, not strength or full-budget convergence.
"""
import json
import os
from pathlib import Path
import socket
import sys
import tomllib
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_placement_runtime_v1 import require_registration, PlacementEpochRandom
from tools.curriculum_placement_parent_v1 import verify_parent
from tools.curriculum_placement_exposures_v1 import REPEAT_FIELD
from tools.curriculum_placement_mix_v1 import masses
from tools.curriculum_placement_objective_v1 import annotate, weighted_loss
from tools.curriculum_placement_metrics_v1 import summarize
from tools.curriculum_checkpoint_evaluation_fence_v1 import CheckpointEvaluationFence, checkpoint_ready

PUBLIC_FILES = ('tools/preflight_curriculum_placement_v1.py',
    'tests/core/test_curriculum_placement_integration_v1.py',
    'cluster/curriculum-placement-preflight-v1.slurm',
    'docs/experiments/plans/curriculum-placement-integration-20261124.md')


def control_rows(rows, fingerprint):
    """One pristine row per source per pool, without shared semantic inputs.

    Both pools are subsets of the CURRENT TRAIN file. The second pool is an
    integration control, not the original held-out validation dataset.
    """
    sources = set(masses()); training = []; validation = []; seen = set(); ids = set()
    for pool in (training, validation):
        covered = set()
        for index, row in enumerate(rows):
            source = row['questions']['action']['src']; meta = row['_meta']
            if source not in sources: raise ValueError('Unknown source in actual TRAIN')
            if source in covered or REPEAT_FIELD in meta: continue
            key = fingerprint(row)
            if key in seen or meta['id'] in ids: continue
            covered.add(source); seen.add(key); ids.add(meta['id']); pool.append((index, row))
        if covered != sources:
            raise ValueError('Each source needs two distinct actual TRAIN control inputs')
    return training, validation


class FilteredControlRandom:
    """Exercise the real full-order adapter, then restrict it to control rows."""
    def __init__(self, full_rng, full_size, selected_indices):
        self.full_rng = full_rng; self.full_size = full_size
        self.selected = {index: position for position, index in enumerate(selected_indices)}
        if len(self.selected) != len(selected_indices): raise ValueError('Distinct control indices required')

    def shuffle(self, order):
        if sorted(order) != list(range(len(self.selected))):
            raise ValueError('Exact admitted control index set required')
        full = list(range(self.full_size)); self.full_rng.shuffle(full)
        restricted = [self.selected[index] for index in full if index in self.selected]
        if sorted(restricted) != list(range(len(order))): raise ValueError('Incomplete filtered actual order')
        order[:] = restricted

    def getstate(self): return self.full_rng.getstate()


def preflight(registered, out):
    gpu_only(); reg = read(registered); require_registration(reg); c = reg['curriculum']
    if out.resolve() == (ROOT/c['out']).resolve() or (ROOT/c['out']).resolve() in out.resolve().parents:
        raise ValueError('Control outputs must be isolated from the student fit')
    if out.exists() or (ROOT/c['out']).exists(): raise ValueError('Never overwrite or run after a student fit starts')
    for path, expected in reg['inputs'].items():
        if sha(path) != expected: raise ValueError('Frozen course prerequisite changed: '+path)
    files = dict(reg['inputs']); files[str(registered.resolve())] = sha(registered)
    for name in PUBLIC_FILES:
        path = (ROOT/name).resolve(); files[str(path)] = sha(path)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevany import train as trainer
    from jevany.suite import semantic_hash
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.initialization import exact_loaded_tensors
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    if (package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']
            or package_digest(Path(reg['training_config']['upstream_source'])/'jevany')
               != reg['training_config']['upstream_source_sha256']):
        raise ValueError('Pinned training packages changed')
    verify_parent(reg['training_config']); parent_before = bundle(reg['parent_model']['directory'])
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    rows = [json.loads(line) for line in Path(args['data']).open()]
    if len(rows) != reg['train_records']: raise ValueError('Actual expanded TRAIN required')
    train, validation = control_rows(rows, semantic_hash)
    if len(train) != 57 or len(validation) != 57: raise ValueError('Complete57-source controls required')
    out.mkdir(parents=True, exist_ok=False)
    for name, selected in (('train-control-private.jsonl', train), ('validation-control-from-TRAIN-private.jsonl', validation)):
        with (out/name).open('x') as stream:
            for _, row in selected: stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    args.update(data=str(out/'train-control-private.jsonl'),
        eval_data=str(out/'validation-control-from-TRAIN-private.jsonl'), out=str(out/'control-run'),
        epochs=2, max_steps=2, batch=1, accum=57, checkpoint_every_steps=1,
        eval_every_steps=1, eval_before_start=False, seed=c['seed'], early_stop_patience=0)
    args.pop('resume', None)
    if Path(args['init_from']).resolve() != Path(reg['parent_model']['directory']).resolve():
        raise ValueError('Actual qualified final3B warm-start required')
    config = out/'control.toml'
    config.write_text(''.join(f'{k} = {json.dumps(v)}\n' for k, v in args.items()))
    categories = [r['questions']['action']['src'] for r in rows]
    full_rng = PlacementEpochRandom(c['seed'], categories, c['accumulation'], out/'full-order-control-private', reg['expected_orders'])
    rng = FilteredControlRandom(full_rng, len(rows), [index for index, _ in train]); created = []
    original = dict(random=trainer.random, materialize=trainer.materialize, loss=trainer.question_loss,
        batch=trainer.batch_loss, tracker=trainer.start_wandb, warm=trainer.Checkpoint.warm_start)
    def factory(value=None):
        if type(value) is int and value == c['seed']:
            if created: raise ValueError('Duplicate control training RNG')
            created.append(rng); return rng
        return original['random'].Random(value)
    trainer.random = SimpleNamespace(seed=original['random'].seed, Random=factory)
    trainer.materialize = lambda request: annotate(original['materialize'](request), request)
    pending = []; journal = []; events = []; inherited = []; optimizer_counts = []
    trainer.question_loss = lambda logits, question, device: weighted_loss(logits, question, device, reg['course_loss_weights'], pending)
    def warm_start(checkpoint, model, meta):
        if Path(checkpoint.path).resolve() != Path(reg['parent_model']['directory']).resolve():
            raise ValueError('Unexpected control initialization checkpoint')
        result = original['warm'](checkpoint, model, meta)
        inherited.append(exact_loaded_tensors(get_peft_model_state_dict(model.lm, save_embedding_layers=False),
            model.head.state_dict(), reg['parent_model']['directory']))
        return result
    trainer.Checkpoint.warm_start = warm_start
    checkpoint_root = out/'control-run/checkpoints'
    def inspect(step):
        state = torch.load(checkpoint_root/f'step-{step:06d}'/'trainer_state.pt', map_location='cpu', weights_only=True)
        if (state['optimizer_step'] != step or len(state['optimizer']['state']) != 376
                or any(float(item['step']) != step for item in state['optimizer']['state'].values())
                or state['args'].get('resume') or state['records_seen'] != 57*step):
            raise ValueError('Actual new376-state optimizer/exposure/save proof failed')
        optimizer_counts.append(dict(step=step, states=376, records_seen=state['records_seen']))
        events.append(dict(kind='callback_after_full_model_and_optimizer_save', step=step))
    fence = CheckpointEvaluationFence(lambda step: checkpoint_ready(checkpoint_root, step), inspect)
    def batch(*a, **kw):
        fence.flush()
        if len(inherited) != 1: raise ValueError('Actual exact inheritance must precede control updates')
        return original['batch'](*a, **kw)
    trainer.batch_loss = batch
    class Tracker:
        config = {}; id = 'placement-integration'; name = id; project = id; entity = None; url = None
        def log(self, values):
            step = values['optimizer_step']
            if 'train/objective' in values:
                actual = summarize(pending, reg['course_loss_weights'])
                if (len(pending) != 57 or set(actual['source_counts']) != set(masses())
                        or abs(actual['loss']-values['train/objective']) > 1e-5):
                    raise ValueError('Actual weighted full57-source objective/journal mismatch')
                journal.append(dict(step=step, **actual, raw_upstream_metrics=values)); pending.clear()
            else:
                events.append(dict(kind='validation_before_save', step=step)); fence.validation(step)
        def finish(self): pass
    trainer.start_wandb = lambda *a, **kw: Tracker()
    try:
        trainer.main(['--config', str(config), '--base_load_path', reg['training_config']['base_load_path']])
        fence.flush()
    finally:
        trainer.random = original['random']; trainer.materialize = original['materialize']
        trainer.question_loss = original['loss']; trainer.batch_loss = original['batch']
        trainer.start_wandb = original['tracker']; trainer.Checkpoint.warm_start = original['warm']
    metrics = read(out/'control-run/training_metrics.json')
    final = checkpoint_root/'step-000002'; parent_path = Path(reg['parent_model']['directory'])
    pa = load_file(parent_path/'adapter_model.safetensors'); ca = load_file(final/'adapter_model.safetensors')
    ph = torch.load(parent_path/'head.pt', map_location='cpu', weights_only=True)['head']
    ch = torch.load(final/'head.pt', map_location='cpu', weights_only=True)['head']
    if (set(pa) != set(ca) or set(ph) != set(ch)
            or not any(not torch.equal(pa[name], ca[name]) for name in pa)
            or not any(not torch.equal(ph[name], ch[name]) for name in ph)
            or metrics['optimizer_steps'] != 2 or metrics['records_seen'] != 114
            or metrics['rejected_records'] or pending or len(journal) != 2
            or fence.finished != {1, 2} or fence.queued != fence.finished
            or full_rng.epochs != 2 or created != [rng]
            or bundle(parent_path) != parent_before):
        raise ValueError('Actual control updates/full orders/isolation/parent preservation failed')
    for path, expected in files.items():
        if sha(path) != expected: raise ValueError('Frozen course input changed during integration')
    write(out/'weighted-loss-journal-private.json', journal)
    write(out/'save-events-private.json', events)
    result = dict(status='actual_placement_upstream_two_update_integration_passed', optimizer_steps=2,
        actual_updates_proven=True, parent_tensors_unchanged=True, isolated_from_student_fit=True,
        actual_weighted_loss_journal_matches=True, saved_optimizer_states_verified=True,
        exact_inheritance=inherited[0], optimizer_checks=optimizer_counts,
        parent_identity=parent_before['identity_sha256'], actual_control_record_exposures=114,
        control_sources=57, complete_prepared_epoch_orders_exercised=2,
        fixtures_from_current_TRAIN_only=True, control_validation_not_original_VAL=True,
        control_distribution_is_not_full_course_mix=True, control_weights_are_not_a_candidate=True,
        strategy_effect_or_native_game_strength_not_evaluated=True,
        fit_launch_admitted=False, automatic_release=False,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
    write(out/'integration-proof-private.json', result)
    paths = [Path(__file__), config, out/'integration-proof-private.json', out/'weighted-loss-journal-private.json',
        out/'save-events-private.json', out/'control-run/training_metrics.json',
        out/'train-control-private.jsonl', out/'validation-control-from-TRAIN-private.jsonl']
    paths += list((out/'full-order-control-private').glob('*.json'))
    paths += [path for directory in checkpoint_root.iterdir() for path in directory.iterdir() if path.is_file()]
    for path in paths: files[str(path.resolve())] = sha(path)
    write(out/'completion-binding-private.json', dict(files=files))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    preflight(a.registered.resolve(), a.out.resolve())
