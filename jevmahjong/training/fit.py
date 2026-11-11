"""Fit Jev Tempo heads on verified native pairs with configurable action balance."""
import argparse
import copy
import json
from pathlib import Path
import random
import torch
from jevmahjong.core.io import digest, write
from jevmahjong.policy.routing import make_head, load_updated_heads
from jevmahjong.evaluation.action_suite import evaluate as action_panel
from jevmahjong.data.native_pairs import load_sources
from .provenance import snapshot, verify
from .rewards import pair_loss
from .sampling import weights, risk_context


@torch.no_grad()
def loss_on(heads, rows, config):
    return sum(float(pair_loss(heads[r['scope']], r, config['kl_beta'], config['pair_beta'],
        config.get('reference_margin_weight', 1.))[0])
        for r in rows) / len(rows)


def initialize_heads(heads, protocol):
    initial = protocol.get('initial_heads')
    if not initial: return {'source': 'reference_heads', 'optimizer_reinitialized': True}
    if digest(Path(initial['checkpoint'])) != initial['sha256']:
        raise ValueError('Initial head checkpoint changed')
    saved = torch.load(initial['checkpoint'], map_location='cpu', weights_only=True)
    if saved.get('base_and_LoRA_frozen') is not True or saved['model_files'] != protocol['model_files']:
        raise ValueError('Initial heads use different frozen encoders')
    for scope, head in heads.items(): head.load_state_dict(saved['heads'][scope])
    return {**initial, 'model_id': saved['model_id'], 'optimizer_reinitialized': True}


def save_head_checkpoint(out, model_id, heads, protocol, epoch, metric):
    checkpoint = out / f'{model_id}-e{epoch:02d}.pt'
    torch.save({'model_id': model_id, 'heads': {s: copy.deepcopy(h.state_dict()) for s, h in heads.items()},
        'model_files': protocol['model_files'], 'base_and_LoRA_frozen': True, 'epoch': epoch,
        'initial_heads': protocol.get('initial_heads')}, checkpoint)
    return {'epoch': epoch, 'held_loss': metric, 'checkpoint': str(checkpoint.resolve()), 'sha256': digest(checkpoint)}


def selection_metric(profile, mode):
    if not profile: raise ValueError('Validation sources required')
    if mode == 'latest_source': return profile[-1]['loss']
    if mode == 'all_sources':
        return sum(p['loss'] * p['rows'] for p in profile) / sum(p['rows'] for p in profile)
    raise ValueError('Unknown validation selection mode')


def validation_profile(heads, rows, sources, config):
    result = []
    for source in sources:
        selected = [r for r in rows if source['seed_start'] <= r['episode_key'][0] < source['seed_start'] + source['seeds']]
        if not selected: raise ValueError('Every source needs held-wall validation')
        contexts = {}
        if config.get('context_mix') or config.get('validation_guard_contexts'):
            for context in ('open_threat', 'other_threat', 'no_threat'):
                subset = [r for r in selected if risk_context(r) == context]
                contexts[context] = {'rows': len(subset),
                    'contrasts': sum(abs(r.get('training_advantage', r['fast_advantage'])) > 1e-12 for r in subset),
                    'loss': loss_on(heads, subset, config) if subset else None}
        result.append({'source': source['path'], 'rows': len(selected), 'loss': loss_on(heads, selected, config),
            **({'risk_contexts': contexts} if contexts else {})})
    return result


def validation_guard(profile, baseline, contexts):
    return all(profile[-1]['risk_contexts'][c]['loss'] is not None
        and baseline[-1]['risk_contexts'][c]['loss'] is not None
        and profile[-1]['risk_contexts'][c]['loss'] <= baseline[-1]['risk_contexts'][c]['loss']
        for c in contexts)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    out = Path(config['out'])
    out.mkdir(parents=True, exist_ok=False)
    protocol = snapshot(config); write(out / 'protocol.json', protocol)
    partitions, hashes, sources = load_sources(config['data'], protocol['model_files'], config.get('reward'))
    train, held = partitions['train'], partitions['held']
    if not held or not train:
        raise ValueError('Whole-wall train and validation partitions required')
    if any(not any(r['scope'] == s and abs(r.get('training_advantage', r['fast_advantage'])) > 1e-12 for r in train)
            for s in ('ordinary', 'calling')):
        write(out / 'result.json', {'status': 'insufficient_reward_contrasts',
            'optimizer_updates': 0, 'deployed': False}); return
    torch.set_num_threads(4); torch.manual_seed(config['seed'])
    heads = {s: make_head(torch.load(Path(config[k]) / 'head.pt', map_location='cpu', weights_only=True), 'cpu')
        for s, k in [('ordinary', 'discard_checkpoint'), ('calling', 'call_checkpoint')]}
    for source in sources:
        reference_heads = (load_updated_heads(source['behavior_heads']['checkpoint'], protocol['model_files'], 'cpu')
            if source.get('behavior_heads') else heads)
        for row in [r for r in train + held if source['seed_start'] <= r['episode_key'][0]
                < source['seed_start'] + source['seeds']]:
            head = reference_heads[row['scope']]
            with torch.no_grad(): p = head(row['decide'], row['options']).softmax(-1)
            if not torch.allclose(p, row['reference'], atol=2e-5, rtol=2e-4):
                raise ValueError('Cached input uses a different source head')
    w, coverage = weights(train, config['action_mix'], config['max_example_weight'], config.get('context_mix'))
    write(out / 'data-coverage.json', coverage)
    initialization = initialize_heads(heads, protocol)
    params = [p for h in heads.values() for p in h.parameters()]
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    mode = config.get('selection_validation', 'all_sources')
    initial_profile = validation_profile(heads, held, sources, config)
    guard = config.get('validation_guard_contexts', [])
    latest = sources[-1]
    fresh_train = [r for r in train if latest['seed_start'] <= r['episode_key'][0] < latest['seed_start'] + latest['seeds']]
    train_contrasts = {c:sum(risk_context(r) == c and abs(r.get('training_advantage', r['fast_advantage'])) > 1e-12
        for r in fresh_train) for c in guard}
    if (any(initial_profile[-1]['risk_contexts'][c]['contrasts'] < config.get('minimum_guard_contrasts', 1) for c in guard)
            or any(train_contrasts[c] < config.get('minimum_train_guard_contrasts', 1) for c in guard)):
        write(out / 'result.json', {'status': 'insufficient_risk_validation_coverage',
            'optimizer_updates': 0, 'initial_validation_sources': initial_profile,
            'latest_source_train_guard_contrasts': train_contrasts, 'deployed': False}); return
    baseline = selection_metric(initial_profile, mode)
    best = save_head_checkpoint(out, config['model_id'], heads, protocol, 0, baseline)
    best['validation_sources'] = initial_profile
    curve = [best]; updates = 0
    stale_epochs = 0; stop_reason = 'epoch_limit'
    for epoch in range(1, config['epochs'] + 1):
        order = list(range(len(train))); random.Random(config['seed'] + epoch).shuffle(order)
        for start in range(0, len(order), config['batch']):
            indices = order[start:start + config['batch']]
            terms = [pair_loss(heads[train[i]['scope']], train[i], config['kl_beta'], config['pair_beta'],
                config.get('reference_margin_weight', 1.))[0] * w[i] for i in indices]
            loss = torch.stack(terms).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite training objective')
            optimizer.zero_grad(); loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(params, .5)
            if not torch.isfinite(gradient): raise ValueError('Nonfinite gradient')
            optimizer.step(); updates += 1
        profile = validation_profile(heads, held, sources, config)
        metric = selection_metric(profile, mode)
        item = save_head_checkpoint(out, config['model_id'], heads, protocol, epoch, metric)
        item['validation_sources'] = profile
        if guard: item['validation_guard_passed'] = validation_guard(profile, initial_profile, guard)
        curve.append(item)
        if metric < best['held_loss'] and validation_guard(profile, initial_profile, guard):
            best = item; stale_epochs = 0
        else:
            stale_epochs += 1
        write(out / 'status.json', {'status': 'training', 'updates': updates, 'curve': curve})
        if config.get('patience_epochs', 0) and stale_epochs >= config['patience_epochs']:
            stop_reason = 'held_loss_stalled'; break
    verify(protocol)
    for path, expected in hashes.items():
        if digest(Path(path)) != expected: raise ValueError('Source data changed during training')
    for source in sources:
        if source.get('behavior_heads'):
            behavior = source['behavior_heads']
            if digest(Path(behavior['checkpoint'])) != behavior['sha256']:
                raise ValueError('Source behavior heads changed during training')
    chosen = torch.load(best['checkpoint'], map_location='cpu', weights_only=True)
    for scope, head in heads.items(): head.load_state_dict(chosen['heads'][scope])
    held_panel = action_panel(heads, held, config['action_cases_per_direction'])
    held_panel['label_target'] = config.get('reward', {'mode': 'fast_win'})
    write(out / 'action-validation.json', held_panel)
    train_panel = action_panel(heads, train, config['action_cases_per_direction'])
    train_panel['role'] = 'train-wall fit diagnostic; not validation or independent strength'
    train_panel['label_target'] = config.get('reward', {'mode': 'fast_win'})
    write(out / 'action-train-diagnostic.json', train_panel)
    result = {'status': 'trained', 'model_id': config['model_id'], 'selected': best, 'curve': curve,
        'baseline_held_loss': baseline, 'optimizer_updates': updates, 'train_rows': len(train), 'held_rows': len(held),
        'stop_reason': stop_reason, 'held_loss_improved': best['held_loss'] < baseline,
        'initialization': initialization, 'selected_is_starting_model': best['epoch'] == 0,
        'selection_validation': mode, 'initial_validation_sources': initial_profile,
        'data': [s['path'] for s in sources] if len(sources) > 1 else sources[0]['path'],
        'data_sources': sources, 'data_sha256': hashes, 'config': config,
        'human_test_opened': False, 'deployed': False, 'strength_improvement_proven': False}
    write(out / 'result.json', result); write(out / 'status.json', result)

if __name__ == '__main__': main()
