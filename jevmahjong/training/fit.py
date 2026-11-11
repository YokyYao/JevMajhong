"""Fit Jev Tempo heads on verified native pairs with configurable action balance."""
import argparse
import copy
import json
from pathlib import Path
import random
import torch
from jevmahjong.core.io import digest, write
from jevmahjong.policy.routing import make_head
from jevmahjong.evaluation.action_suite import evaluate as action_panel
from .collect import validate
from .provenance import snapshot, verify
from .rewards import pair_loss
from .sampling import weights


@torch.no_grad()
def loss_on(heads, rows, config):
    return sum(float(pair_loss(heads[r['scope']], r, config['kl_beta'], config['pair_beta'],
        config.get('reference_margin_weight', 1.))[0])
        for r in rows) / len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True)
    config = json.loads(parser.parse_args().config.read_text())
    data, out = Path(config['data']), Path(config['out'])
    collection = json.loads((data / 'collection-result.json').read_text())
    source = json.loads((data / 'protocol.json').read_text())['config']
    if collection['status'] != 'collection_complete':
        raise ValueError('Complete native collection required')
    out.mkdir(parents=True, exist_ok=False)
    protocol = snapshot(config); write(out / 'protocol.json', protocol)
    rows = []; hashes = {}
    for path in sorted((data / 'walls-private').glob('*.pt')):
        h = digest(path)
        if h != json.loads(path.with_suffix('.audit.json').read_text())['sha256']:
            raise ValueError('Source shard changed')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        validate(payload, payload['seed'], source)
        rows.extend(payload['rows']); hashes[str(path.resolve())] = h
    if len(hashes) != collection['walls'] or len(rows) != collection['rows']:
        raise ValueError('Source collection coverage differs')
    held = [r for r in rows if (r['episode_key'][0] - source['training_seed_start']) % 4 == 3]
    train = [r for r in rows if (r['episode_key'][0] - source['training_seed_start']) % 4 != 3]
    if not held or not train:
        raise ValueError('Whole-wall train and validation partitions required')
    if any(not any(r['scope'] == s and abs(r['fast_advantage']) > 1e-12 for r in train)
            for s in ('ordinary', 'calling')):
        write(out / 'result.json', {'status': 'insufficient_reward_contrasts',
            'optimizer_updates': 0, 'deployed': False}); return
    torch.set_num_threads(4); torch.manual_seed(config['seed'])
    heads = {s: make_head(torch.load(Path(config[k]) / 'head.pt', map_location='cpu', weights_only=True), 'cpu')
        for s, k in [('ordinary', 'discard_checkpoint'), ('calling', 'call_checkpoint')]}
    for scope, head in heads.items():
        for row in [r for r in train if r['scope'] == scope][:20]:
            with torch.no_grad(): p = head(row['decide'], row['options']).softmax(-1)
            if not torch.allclose(p, row['reference'], atol=2e-5, rtol=2e-4):
                raise ValueError('Cached input uses a different source head')
    w, coverage = weights(train, config['action_mix'], config['max_example_weight'])
    write(out / 'data-coverage.json', coverage)
    params = [p for h in heads.values() for p in h.parameters()]
    optimizer = torch.optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    baseline = loss_on(heads, held, config); best = None; curve = []; updates = 0
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
        metric = loss_on(heads, held, config)
        checkpoint = out / f"{config['model_id']}-e{epoch:02d}.pt"
        torch.save({'model_id': config['model_id'], 'heads': {s: copy.deepcopy(h.state_dict()) for s, h in heads.items()},
            'model_files': protocol['model_files'], 'base_and_LoRA_frozen': True, 'epoch': epoch}, checkpoint)
        item = {'epoch': epoch, 'held_loss': metric, 'checkpoint': str(checkpoint.resolve()), 'sha256': digest(checkpoint)}
        curve.append(item)
        if best is None or metric < best['held_loss']:
            best = item; stale_epochs = 0
        else:
            stale_epochs += 1
        write(out / 'status.json', {'status': 'training', 'updates': updates, 'curve': curve})
        if config.get('patience_epochs', 0) and stale_epochs >= config['patience_epochs']:
            stop_reason = 'held_loss_stalled'; break
    verify(protocol)
    chosen = torch.load(best['checkpoint'], map_location='cpu', weights_only=True)
    for scope, head in heads.items(): head.load_state_dict(chosen['heads'][scope])
    write(out / 'action-validation.json', action_panel(heads, held, config['action_cases_per_direction']))
    train_panel = action_panel(heads, train, config['action_cases_per_direction'])
    train_panel['role'] = 'train-wall fit diagnostic; not validation or independent strength'
    write(out / 'action-train-diagnostic.json', train_panel)
    result = {'status': 'trained', 'model_id': config['model_id'], 'selected': best, 'curve': curve,
        'baseline_held_loss': baseline, 'optimizer_updates': updates, 'train_rows': len(train), 'held_rows': len(held),
        'stop_reason': stop_reason, 'held_loss_improved': best['held_loss'] < baseline,
        'data': str(data.resolve()), 'data_sha256': hashes, 'config': config,
        'human_test_opened': False, 'deployed': False, 'strength_improvement_proven': False}
    write(out / 'result.json', result); write(out / 'status.json', result)

if __name__ == '__main__': main()
