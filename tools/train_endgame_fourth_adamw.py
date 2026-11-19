"""Bounded full-source AdamW ablation of the short, projected fourth-place fit.

Reuses round16 labels, masses and exact v8 encoder features. Validation controls
selection/early stopping only; no validation gradient enters the optimizer.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_config(config):
    expected = {
        'schema': 'endgame-fourth-full-source-adamw-v1',
        'out': 'runs/endgame-fourth-adamw-v1',
        'prepared_data_run': 'runs/endgame-fourth-top2-chase-v1',
        'training_feature_cache_run': 'runs/endgame-head-gradient-diagnostic-v1',
        'validation_feature_cache_run': 'runs/endgame-placement-v15',
        'parent': 'released_v8_step2088', 'trainable_scope': 'existing_head_only',
        'trainable_parameters': 524800, 'optimizer': 'AdamW',
        'learning_rate': .00005, 'weight_decay': .001, 'gradient_clip_norm': 1.,
        'learning_rate_warmup_steps': 8,
        'learning_rate_schedule': 'linear_warmup_then_constant',
        'max_optimizer_steps': 128, 'validation_every_steps': 8,
        'early_stop_observation_start': 32, 'early_stop_patience': 4,
        'soft_progress_min_delta': .00001,
        'fourth_escape_constraint': 'report_only',
        'fourth_native_kl_constraint': 'report_only',
        'selection_guard_protocol': 'unchanged_from_round16',
        'per_source_ce_monotonicity': False, 'severe_replay_ce_increase_stop': .5,
        'fit_registered': False, 'independent_confirmation': False,
        'export_registered': False, 'raw_publication': False,
    }
    if config != expected:
        raise ValueError('Only the preregistered full-source AdamW ablation is enabled')


def progress_state(step, metrics, guards, previous, config):
    """Allow a learning window; keep selection distinct from training patience."""
    hard, soft = metrics['fourth']['top2'], metrics['fourth']['soft_top2']
    tail = metrics['fourth']['fourth']
    if not all(math.isfinite(x) for x in (hard, soft, tail)):
        raise ValueError('Nonfinite validation metric')
    if step == 0:
        return dict(selected_step=0, selected_hard=None, selected_fourth=None,
                    best_hard=hard, best_soft=soft, stale=0, stop=False)
    state = dict(previous)
    selected = guards['eligible_nonzero'] and (
        state['selected_hard'] is None or hard > state['selected_hard'] + 1e-12
        or abs(hard-state['selected_hard']) <= 1e-12 and tail < state['selected_fourth']-1e-12)
    if selected:
        state.update(selected_step=step, selected_hard=hard, selected_fourth=tail)
    advancing = (selected or hard > state['best_hard']+1e-12
                 or soft > state['best_soft']+config['soft_progress_min_delta'])
    state['best_hard'] = max(state['best_hard'], hard)
    state['best_soft'] = max(state['best_soft'], soft)
    if step <= config['early_stop_observation_start']:
        state['stale'] = 0
    else:
        state['stale'] = 0 if advancing else state['stale']+1
    state['stop'] = state['stale'] >= config['early_stop_patience']
    return state


def combine_source_gradients(gradients, masses):
    import torch
    if set(gradients) != set(masses) or abs(sum(masses.values())-1.) > 1e-9:
        raise ValueError('Every positive training source must contribute its fixed mass')
    result = sum(masses[k]*gradients[k].double() for k in gradients)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite full training gradient')
    return result


def scheduled_lr(step, config):
    if not 1 <= step <= config['max_optimizer_steps']:
        raise ValueError('Step outside registered schedule')
    return config['learning_rate']*min(1., step/config['learning_rate_warmup_steps'])


def validate_assets(config):
    """Reuse immutable, already audited targets without rebuilding or relabeling."""
    from tools.train_endgame_fourth_chase import verify
    from jevmahjong.endgame.prepare import checksum, read_jsonl
    from jevmahjong.endgame.futures import bundle_identity
    root = Path(config['prepared_data_run'])
    ready = json.loads((root/'ready-private.json').read_text())
    result = json.loads((root/'status-private.json').read_text())
    if (ready['status'] != 'fourth_top2_chase_ready_fit_disabled'
            or result['status'] != 'fourth_top2_chase_fit_complete'
            or result['last_step'] != 2 or result['selected_step'] != 0):
        raise ValueError('Expected the completed unqualified round16 source')
    verify(json.loads((root/'prepare-protocol-private.json').read_text()))
    for path, expected in ready['data_sha256'].items():
        if checksum(path) != expected:
            raise ValueError('Prepared training or validation data changed')
    if checksum(root/'reference-targets-private.json') != ready['reference_sha256']:
        raise ValueError('Frozen references changed')
    parent = Path(ready['source_config']['checkpoint'])
    if bundle_identity(parent) != ready['parent_identity']:
        raise ValueError('Formal v8 parent changed')
    all_train = read_jsonl(root/'train.jsonl')
    validation = read_jsonl(root/'validation.jsonl')
    recipe = ready['loss_recipe']
    train = [r for r in all_train if recipe['weights'][r['questions']['action']['src']] > 0]
    fourth = [r for r in train if 'fourth_chase_top2_probabilities' in r['_meta']]
    if (len(all_train) != 905 or len(train) != 826 or len(fourth) != 149
            or len(validation) != 186
            or {r['_meta']['id'] for r in all_train} & {r['_meta']['id'] for r in validation}):
        raise ValueError('Fixed membership or split isolation changed')
    return ready, parent, train, validation, fourth


def fit(config, registered):
    validate_config(config)
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.prepare import checksum
    from tools.train_endgame_fourth_chase import verify, read_caches
    from tools.train_endgame_constrained_head import head_predictions
    from tools.audit_endgame_head_gradients import source_gradients, gradient_directions
    from tools.audit_endgame_parameter_attribution import control_comparison
    from jevmahjong.endgame.constrained_head import updated_head_metadata
    from jevmahjong.endgame.fourth_chase_training import chase_validation, chase_guards, chase_training_metrics
    out = Path(config['out'])
    if any((out/name).exists() for name in ('protocol-private.json', 'training-run', 'fit-result-private.json')):
        raise ValueError('Inspect existing run; never silently restart it')
    if registered['configuration'] != config or registered['ci']['conclusion'] != 'success':
        raise ValueError('Missing successful public source registration')
    verify(registered)
    ready, parent, train, validation, fourth = validate_assets(config)
    train_cache, val_cache = read_caches(config, ready['parent_identity'], train, validation)
    sys.path.insert(0, ready['source_config']['upstream_source'])
    import torch
    from jevany.model import PointerHead
    torch.manual_seed(2026111907)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = torch.load(parent/'head.pt', map_location='cpu', weights_only=True)
    dp, d = meta['head']['q.weight'].shape
    head = PointerHead(d, dp).to(device='cuda', dtype=torch.float32).requires_grad_(True)
    head.load_state_dict(meta['head'], strict=True)
    parameters = list(head.parameters())
    if sum(p.numel() for p in parameters) != 524800:
        raise ValueError('Existing pointer head architecture changed')
    initial_flat = torch.cat([p.detach().reshape(-1) for p in parameters]).double().clone()
    features = dict(train_cache['features'], **val_cache['features'])
    for value in features.values():
        value['decide'] = value['decide'].to('cuda').clone()
        value['options'] = value['options'].to('cuda').clone()
    controls = {}
    for name, rows, cache in (('train826', train, train_cache), ('validation186', validation, val_cache)):
        controls[name] = control_comparison(rows, head_predictions(head, rows, features), cache['actual_probabilities'], 1e-7)
        if not controls[name]['passed'] or controls[name]['max_absolute_probability_difference'] != 0.:
            raise ValueError('Zero head differs from audited actual service probabilities')
    source_root = Path(config['prepared_data_run'])
    reference = json.loads((source_root/'reference-targets-private.json').read_text())['rows']
    recipe = ready['loss_recipe']
    masses = {k: v for k, v in recipe['mass'].items() if v > 0}
    protocol = dict(registered, parent_identity=ready['parent_identity'], zero_controls=controls,
                    source_ready=checksum(source_root/'ready-private.json'),
                    data_sha256=ready['data_sha256'], reference_sha256=ready['reference_sha256'])
    write(out/'protocol-private.json', protocol)
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    root = out/'training-run'
    curve, updates, progress = [], [], None
    initial = ready['initial_validation']
    stop_reason = 'registered_max128_updates'

    def evaluate(step):
        nonlocal progress
        predictions = head_predictions(head, validation, features)
        metrics = chase_validation(validation, predictions, reference)
        if step == 0 and metrics != initial:
            raise ValueError('New trial changed the round16 initial validation anchor')
        guards = chase_guards(step, metrics, initial, ready['source_config'], exposure_met=step > 0)
        progress = progress_state(step, metrics, guards, progress, config)
        point = dict(step=step, metrics=metrics, guards=guards, progress=progress,
                     training149=chase_training_metrics(fourth, head_predictions(head, fourth, features), reference),
                     full_positive_training_passes=step, full_positive_exposure=826 if step else 0,
                     cumulative_positive_presentations=step*826)
        curve.append(point)
        path = root/'checkpoints'/f'step-{step:06d}'
        shutil.copytree(parent, path, ignore=shutil.ignore_patterns('trainer_state*'))
        saved = updated_head_metadata(meta, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, step)
        saved.update(training_backend='full_source_adamw_fourth_chase', policy_version='endgame-fourth-chase-v1')
        torch.save(saved, path/'head.pt')
        torch.save({'optimizer': optimizer.state_dict(), 'step': step}, path/'trainer_state.pt')
        write(root/'predictions'/f'step-{step:06d}-private.json', predictions)
        write(root/'selection.json', dict(step=progress['selected_step'],
              selected_checkpoint=f"checkpoints/step-{progress['selected_step']:06d}",
              metric='fourth_final_top2_with_unchanged_third_leading_selection_guards'))
        write(out/'progress-private.json', point)
        write(out/'curve-private.json', curve)
        print(json.dumps({'validation_step': step, 'fourth': metrics['fourth'],
                          'common_guards_passed': guards['common_guards_passed'], 'progress': progress,
                          'training149': point['training149']}), flush=True)
        return point

    evaluate(0)
    for step in range(1, config['max_optimizer_steps']+1):
        head.train()
        gradients, losses = source_gradients(head, train, features)
        full = combine_source_gradients(gradients, masses)
        if step == 1:
            diagnostics, _ = gradient_directions(gradients, masses, initial_flat)
            # Descriptive training gradients only; never optimize on validation.
            write(out/'initial-training-gradients-private.json', diagnostics)
        optimizer.zero_grad(set_to_none=True)
        offset = 0
        for parameter in parameters:
            parameter.grad = full[offset:offset+parameter.numel()].reshape_as(parameter).to(parameter.dtype).clone()
            offset += parameter.numel()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip_norm'], error_if_nonfinite=True)
        lr = scheduled_lr(step, config)
        for group in optimizer.param_groups:
            group['lr'] = lr
        before = torch.cat([p.detach().reshape(-1) for p in parameters]).double().clone()
        optimizer.step()
        after = torch.cat([p.detach().reshape(-1) for p in parameters]).double()
        if not torch.isfinite(after).all():
            raise ValueError('Nonfinite updated parameters')
        update = dict(step=step, lr=lr, source_ce_before=losses,
                      weighted_ce_before=sum(masses[k]*losses[k] for k in masses),
                      gradient_norm_before_clip=float(grad_norm),
                      relative_head_l2_change=float(torch.linalg.vector_norm(after-before)/torch.linalg.vector_norm(before)),
                      relative_head_l2_from_parent=float(torch.linalg.vector_norm(after-initial_flat)/torch.linalg.vector_norm(initial_flat)),
                      positive_records=826, per_source_monotonic_veto=False)
        updates.append(update)
        write(out/'updates-private.json', updates)
        print(json.dumps(update), flush=True)
        if step % config['validation_every_steps'] == 0 or step == config['max_optimizer_steps']:
            point = evaluate(step)
            if point['metrics']['legacy']['original_expert_ce'] > initial['legacy']['original_expert_ce']+config['severe_replay_ce_increase_stop']:
                stop_reason = 'severe_original_replay_ce_regression'; break
            if progress['stop']:
                stop_reason = 'four_stale_validations_after32_observation_updates'; break
    verify(registered)
    validate_assets(config)
    final_losses = source_gradients(head, train, features)[1]
    result = dict(status='fourth_full_source_adamw_fit_complete', last_step=updates[-1]['step'],
                  selected_step=progress['selected_step'], eligible_for_service_audit=progress['selected_step'] > 0,
                  stop_reason=stop_reason, curve=curve, updates=updates,
                  initial_source_ce=updates[0]['source_ce_before'], final_source_ce=final_losses,
                  full_positive_training_passes=len(updates), unique_positive_records=826,
                  training_encoder_queries=0, new_labels=0, external_networks=0,
                  fourth_escape_is_not_a_veto=True, deployed=False, raw_publication=False)
    write(out/'fit-result-private.json', result)
    write(out/'status-private.json', {k:v for k,v in result.items() if k not in ('curve','updates','initial_source_ce','final_source_ce')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-', 'login')):
        raise RuntimeError('Only scheduler-allocated GPU computation is permitted')
    from tools.prepare_native_context_replay import allocation_metadata
    description = subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True)
    print(json.dumps({'node':socket.gethostname(), 'allocation':allocation_metadata(description,socket.gethostname())}),flush=True)
    fit(json.loads(args.config.read_text()), json.loads(args.registration.read_text()))


if __name__ == '__main__':
    main()
