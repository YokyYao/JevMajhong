"""Train both hybrid action heads from same-hand reward preferences, not expert CE."""
import argparse
import copy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from call_policy import digest
from fast_win import pair_loss
from head_rl import write
from scripts.collect_fast_win import validate, verify_source
from scripts.train_call_repair import make_head


@torch.no_grad()
def evaluate(heads, rows, cfg):
    values = [float(pair_loss(heads[r['scope']], r, cfg['kl_beta'], cfg['pair_beta'])[0]) for r in rows]
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True)
    cfg = json.loads(parser.parse_args().config.read_text()); out = ROOT / cfg['out']
    protocol = json.loads((out / 'protocol.json').read_text()); verify_source(protocol, cfg)
    collection = json.loads((out / 'collection-result.json').read_text())
    if collection['status'] != 'collection_complete' or (out / 'fit-result.json').exists():
        raise ValueError('New complete collection required')
    torch.set_num_threads(4); torch.manual_seed(cfg['seed'])
    rows = []; shards = {}
    for seed in range(cfg['training_seed_start'], cfg['training_seed_start'] + cfg['training_seeds_per_stage']):
        shard = out / 'walls-private' / f'{seed}.pt'
        h = digest(shard)
        if h != json.loads(shard.with_suffix('.audit.json').read_text())['sha256']:
            raise ValueError('Training shard changed')
        payload = torch.load(shard, map_location='cpu', weights_only=False); validate(payload, seed, cfg)
        rows.extend(payload['rows']); shards[str(shard.relative_to(ROOT))] = h
    if len(rows) != collection['rows']:
        raise ValueError('Training collection count differs')
    # Last residue of each four-wall group is held; all seats/replicas stay together.
    held = [r for r in rows if (r['episode_key'][0] - cfg['training_seed_start']) % 4 == 3]
    train = [r for r in rows if (r['episode_key'][0] - cfg['training_seed_start']) % 4 != 3]
    if not held or not train or any(not any(r['scope'] == s and abs(r['fast_advantage']) > 1e-12 for r in train)
        for s in ('ordinary', 'calling')):
        write(out / 'fit-result.json', {'status': 'insufficient_reward_contrasts',
            'rows': len(rows), 'trained': False, 'deployed': False, 'requires_more_native_training_walls': True})
        return
    heads = {}
    for scope, key in [('ordinary', 'reference'), ('calling', 'teacher_checkpoint')]:
        meta = torch.load(ROOT / cfg[key] / 'head.pt', map_location='cpu', weights_only=True)
        heads[scope] = make_head(meta, 'cpu').train().requires_grad_(True)
        # Feature capture must reproduce its own frozen source before changing either head.
        for row in [r for r in rows if r['scope'] == scope][:20]:
            with torch.no_grad():
                p = heads[scope](row['decide'], row['options']).softmax(-1)
            if not torch.allclose(p, row['reference'], atol=2e-5, rtol=2e-4):
                raise ValueError('Wrong cached representation for ' + scope)
    parameters = [p for head in heads.values() for p in head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=cfg['head_learning_rate'], weight_decay=cfg['weight_decay'])
    baseline = evaluate(heads, held, cfg); curve = []; best = None; updates = 0
    for epoch in range(1, cfg['head_epochs'] + 1):
        order = list(range(len(train))); random.Random(cfg['seed'] + epoch).shuffle(order)
        for start in range(0, len(order), cfg['head_batch_size']):
            batch = [train[i] for i in order[start:start + cfg['head_batch_size']]]
            loss = torch.stack([pair_loss(heads[r['scope']], r, cfg['kl_beta'], cfg['pair_beta'])[0] for r in batch]).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite reward preference loss')
            optimizer.zero_grad(); loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(parameters, .5)
            if not torch.isfinite(gradient):
                raise ValueError('Nonfinite policy gradient')
            optimizer.step(); updates += 1
        held_loss = evaluate(heads, held, cfg)
        item = {'epoch': epoch, 'updates': updates, 'held_reward_preference_loss': held_loss}
        curve.append(item)
        checkpoint = out / f'fast-heads-epoch-{epoch:02d}-private.pt'
        torch.save({'heads': {s: copy.deepcopy(h.state_dict()) for s, h in heads.items()},
            'config': cfg, 'teacher_files': protocol['teacher_files'],
            'reference_files': protocol['reference_files'], 'epoch': epoch,
            'method': 'same_hand_fast_reward_weighted_reference_relative_pairwise_plus_KL',
            'base_and_LoRA_frozen': True, 'all_three_kans_use_ordinary_head': True}, checkpoint)
        item['checkpoint_sha256'] = digest(checkpoint)
        if best is None or held_loss < best['held_reward_preference_loss']:
            best = dict(item, checkpoint=str(checkpoint.relative_to(ROOT)))
        write(out / 'status.json', {'status': 'fitting_fast_win_heads', 'curve': curve,
            'baseline_held_loss': baseline, 'optimizer_updates': updates})
    verify_source(protocol, cfg)
    result = {'status': 'fast_heads_trained', 'optimizer_updates': updates,
        'trainable_parameters': sum(p.numel() for p in parameters),
        'train_requests': len(train), 'held_requests': len(held),
        'train_walls': len({r['episode_key'][0] for r in train}),
        'held_walls': len({r['episode_key'][0] for r in held}),
        'baseline_held_loss': baseline, 'curve': curve, 'selected': best,
        'training_shards': shards, 'base_and_LoRA_updated': False,
        'selection': 'Lowest held whole-wall reward preference loss among updated epochs; needs fresh full games',
        'strength_improvement_proven': False, 'human_test_opened': False, 'deployed': False}
    write(out / 'fit-result.json', result); write(out / 'status.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'training_shards'}), flush=True)


if __name__ == '__main__':
    main()
