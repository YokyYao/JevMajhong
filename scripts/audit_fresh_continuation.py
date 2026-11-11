"""Recompute saved validation, original NLL selection, schedule and data exposure."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from checkpoint_identity import completed_checkpoint_files
from feature_continuation import verify_preparation
from head_rl import write


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    prepared = json.loads((out / 'preparation.json').read_text()); verify_preparation(ROOT, cfg, prepared)
    result = json.loads((out / 'result.json').read_text())
    metrics = json.loads((out / 'training-run/training_metrics.json').read_text())
    records = [json.loads(l) for l in (ROOT / cfg['data'] / 'validation.jsonl').open()]
    lookup = {r['_meta']['id']: r for r in records}
    if len(lookup) != len(records): raise ValueError('Duplicate validation identities')
    curve = json.loads((out / 'validation-curve.json').read_text())
    expected_steps = list(range(0, result['optimizer_steps'] + 1, cfg['validation_every']))
    if [r['step'] for r in curve] != expected_steps: raise ValueError('Saved validation schedule incomplete')
    table = []; prediction_count = 0
    for item in curve:
        step = item['step']; rows = json.loads((out / 'training-run/training_eval' /
            f'step-{step:06d}/development/rows.json').read_text())
        index = {r['id']: r for r in rows}
        if len(index) != len(rows) or set(index) != set(lookup): raise ValueError('Saved prediction coverage differs')
        loss = []; phase_loss = {}; support = Counter(); exact = Counter()
        for identity, r in index.items():
            record = lookup[identity]; question = record['questions']['action']; keys = list(question['criteria'])
            if r['keys'] != keys or r['label'] != keys.index(question['label']):
                raise ValueError('Prediction/target option alignment differs')
            probability = r['p']
            if (len(probability) != len(keys) or any(not math.isfinite(v) or v < 0 for v in probability)
                    or abs(sum(probability) - 1.) > 1e-5): raise ValueError('Invalid saved distribution')
            gold = keys.index(question['label']); kind = question['criteria'][question['label']]['type']
            value = -math.log(max(probability[gold], 1e-30)); loss.append(value)
            phase_loss.setdefault(record['_meta']['phase'], []).append(value)
            support[kind] += 1
            if max(range(len(probability)), key=lambda i: probability[i]) == gold: exact[kind] += 1
        nll = sum(loss) / len(loss); saved = item['views']['native']
        if (abs(nll - item['nll']) > 2e-7 or dict(support) != saved['support']
                or dict(exact) != saved['exact_tp']): raise ValueError('Saved validation aggregate differs')
        for phase, values in phase_loss.items():
            if abs(sum(values) / len(values) - item['by_phase'][phase]['native']['nll']) > 2e-7:
                raise ValueError('Phase validation differs')
        admissible = exact['chi'] > 0 and exact['pon'] > 0
        if item['guard']['admissible'] != admissible: raise ValueError('Original minimum-skill admission differs')
        table.append({'step': step, 'full_nll': item['nll'],
            'reaction_nll': sum(phase_loss['reaction']) / len(phase_loss['reaction']),
            'exact_chi': exact['chi'], 'exact_pon': exact['pon'], 'expert_pass_disagreements': support['none'] - exact['none'],
            'admissible': admissible})
        prediction_count += len(rows)
    updated = [r for r in table if r['step'] > 0 and r['admissible']]
    best = min(updated, key=lambda r: (r['full_nll'], r['step'])) if updated else None
    if (best['step'] if best else None) != result['selected_step']:
        raise ValueError('Original training NLL selection differs')
    train = [json.loads(l) for l in (ROOT / cfg['data'] / 'train.jsonl').open()]
    exposure = {}; common_order = None
    for step in expected_steps[1:]:
        checkpoint = out / 'training-run/checkpoints' / f'step-{step:06d}'
        completed_checkpoint_files(checkpoint, step)
        state = torch.load(checkpoint / 'trainer_state.pt', map_location='cpu', weights_only=True)
        order = state['order'].tolist(); count = state['records_seen']
        if (state['epoch'] != 0 or state['next_microbatch'] != count or count != step * 16
                or state['world_size'] != 1 or state['args']['batch'] != 1
                or len(order) != len(train) or sorted(order) != list(range(len(train)))):
            raise ValueError('Actual single-epoch training cursor/exposure differs')
        if common_order is not None and order != common_order: raise ValueError('Checkpoint order changed mid-epoch')
        common_order = order
        scheduler = state['scheduler']
        if scheduler['total_steps'] != cfg['max_steps'] or scheduler['last_epoch'] != step:
            raise ValueError('OneCycle budget differs from registered200step recipe')
        subset = [train[i] for i in order[:count]]
        ids = [r['_meta']['id'] for r in subset]
        if len(set(ids)) != count: raise ValueError('Repeated TRAIN exposure inside first epoch')
        phases = Counter(r['_meta']['phase'] for r in subset)
        actions = Counter(r['questions']['action']['criteria'][r['questions']['action']['label']]['type'] for r in subset)
        exposure[str(step)] = {'unique_requests': count, 'phases': dict(phases), 'actions': dict(actions),
            'epoch': state['epoch'], 'next_microbatch': state['next_microbatch'],
            'scheduler_total_steps': scheduler['total_steps']}
    if (metrics['optimizer_steps'] != result['optimizer_steps'] or metrics['records_seen'] != result['optimizer_steps'] * 16
            or metrics['global_effective_batch'] != 16 or metrics['rejected_records'] or metrics['early_stopped']):
        raise ValueError('Final actual training metrics differ')
    report = {'status': 'passed_complete_training_audit', 'prediction_rows_recomputed': prediction_count,
        'optimizer_steps': metrics['optimizer_steps'], 'records_seen': metrics['records_seen'],
        'pool_records': len(train), 'train_matches': prepared['train_matches'],
        'selected_by_original_nll': result['selected_step'], 'curve': table, 'exposure': exposure,
        'scheduler_cap_verified': cfg['max_steps'],
        'uncapped_dataset_optimizer_steps': metrics['planned_optimizer_steps'],
        'no_new_optimizer_updates': True, 'human_test_opened': False, 'strength_improvement_proven': False}
    write(out / 'training-completion-audit.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
