"""Fit a small action-value correction from paired terminal points, not guide labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from call_policy import digest
from diagonal_value import cross_validate, pointer_products, ridge_fit
from guide_call_features import VERSION, NAMES
from head_rl import write
from scripts.collect_guide_call_points import validate_wall, CODE
from scripts.train_call_repair import make_head


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    collection = json.loads((out / 'collection-result.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    if collection['status'] != 'collection_complete' or protocol['config'] != cfg: raise ValueError('Collection incomplete')
    for name in CODE:
        if digest(ROOT / name) != protocol['code_sha256'][name]: raise ValueError('Collection source changed')
    if (out / 'fit-result.json').exists(): raise ValueError('Fit already recorded')
    meta = torch.load(ROOT / cfg['reference'] / 'head.pt', map_location='cpu', weights_only=True)
    call = torch.load(ROOT / cfg['call_checkpoint'], map_location='cpu', weights_only=True)
    if digest(ROOT / cfg['call_checkpoint']) != cfg['call_head_sha256']: raise ValueError('Teacher changed')
    head = make_head(meta, 'cpu'); head.load_state_dict(call['head']); head.eval().requires_grad_(False)
    xs, ys, seeds = [], [], []; total = 0
    for seed in range(cfg['training_seed_start'], cfg['training_seed_start'] + cfg['training_seeds_per_stage']):
        shard = out / 'walls-private' / f'{seed}.pt'
        if digest(shard) != json.loads(shard.with_suffix('.audit.json').read_text())['sha256']:
            raise ValueError('Shard changed')
        payload = torch.load(shard, map_location='cpu', weights_only=False); validate_wall(payload, seed, cfg)
        games = {(g['seed'], g['seat'], g['replica']):g for g in payload['games']}
        for row in payload['rows']:
            total += 1; action, teacher = row['action'], row['teacher_action']
            delta = games[tuple(row['episode_key'])]['terminal_point_delta']
            if action == teacher:
                if delta != 0: raise ValueError('Nonzero same-action label')
                continue  # Do not dilute CV MSE with known zero-versus-zero cases.
            with torch.no_grad(): latent = pointer_products(head, row['decide'], row['options'])
            features = torch.cat((latent, row['strategy_features']), dim=1)
            xs.append(features[action] - features[teacher]); ys.append(delta / cfg['reward_scale']); seeds.append(seed)
    if len(set(seeds)) < cfg['cv_folds']:
        write(out / 'fit-result.json', {'status':'insufficient_wall_coverage','paired_rows':total,
            'changed_action_rows':len(ys),'trained':False,'deployed':False}); return
    x, y = torch.stack(xs), torch.tensor(ys, dtype=torch.float64)
    validation = cross_validate(x, y, seeds, cfg['ridge_alphas'], folds=cfg['cv_folds'])
    result = {'status':'point_predictor_fit' if validation['improves_over_zero'] else 'point_predictor_not_improved',
        'paired_rows':total,'changed_action_rows':len(ys),'independent_training_walls':len(set(seeds)),
        'feature_version':VERSION,'guide_feature_count':len(NAMES),'total_features':x.shape[1],
        'validation':validation,'selection':'whole-wall grouped CV MSE versus predicting zero action difference',
        'reward':'unchanged full-East terminal net points /1000; no hand-crafted call, han or rank bonus',
        'deployed':False,'human_test_opened':False,'strength_improvement_proven':False,
        'requires_new_full_match_development':True}
    if validation['improves_over_zero']:
        coef = ridge_fit(x, y, validation['best_alpha'])
        checkpoint = out / 'point-value-private.pt'
        torch.save({'coefficient':coef, 'config':cfg,'strategy_feature_version':VERSION,
            'strategy_feature_names':list(NAMES),'teacher_sha256':cfg['call_head_sha256'],
            'reference_sha256':cfg['reference_head_sha256'],'target_scale':cfg['reward_scale'],
            'scope':['chi','pon','none'],'validation':validation}, checkpoint)
        result['checkpoint_sha256'] = digest(checkpoint)
    write(out / 'fit-result.json', result); write(out / 'status.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
