"""Training-only trajectory score gradients and cross-stream weighted control baselines."""
import math
import statistics

import torch
from head_rl import mixed_probabilities


def trajectory_score_norm(head, rows, exploration, verify_cache=False):
    head.zero_grad(set_to_none=True)
    device = next(head.parameters()).device
    values = []
    for row in rows:
        logits = head(row['decide'].to(device), row['options'].to(device))
        probability = mixed_probabilities(logits, row['eligible'], exploration)
        logp = probability[row['action']].log()
        if verify_cache:
            if not torch.allclose(probability.detach().cpu(), row['reference'], atol=2e-5, rtol=2e-4):
                raise ValueError('Gradient diagnostic behavior probabilities differ')
            if abs(float(logp.detach()) - row['old_logp']) > 2e-4:
                raise ValueError('Gradient diagnostic action likelihood differs')
        values.append(logp)
    if not values:
        return 0.0
    torch.stack(values).sum().backward()
    norm = sum(float(p.grad.detach().float().square().sum()) for p in head.parameters() if p.grad is not None)
    head.zero_grad(set_to_none=True)
    if not math.isfinite(norm):
        raise ValueError('Nonfinite trajectory score norm')
    return norm


def score_weighted_baselines(games, controls, weights):
    control = {(g['seed'], g['seat']): float(g['reward']) for g in controls}
    keys = [(g['seed'], g['seat'], g['replica']) for g in games]
    if len(control) != len(controls) or len(set(keys)) != len(keys):
        raise ValueError('Duplicate training episode or control')
    seeds = sorted({g['seed'] for g in games})
    if len(seeds) < 3:
        raise ValueError('Need at least three independent streams')
    rr = [float(g['reward']) for g in games]
    cc = [control[g['seed'], g['seat']] for g in games]
    ww = [float(weights[k]) for k in keys]
    if not all(math.isfinite(x) for x in rr + cc + ww) or any(x < 0 for x in ww) or sum(ww) <= 0:
        raise ValueError('Invalid score-gradient weights or returns')
    baselines, fits = {}, []
    for seed in seeds:
        ix = [i for i, g in enumerate(games) if g['seed'] != seed]
        total = sum(ww[i] for i in ix)
        if total > 0:
            mr = sum(ww[i] * rr[i] for i in ix) / total
            mc = sum(ww[i] * cc[i] for i in ix) / total
            variance = sum(ww[i] * (cc[i] - mc)**2 for i in ix)
            covariance = sum(ww[i] * (rr[i] - mr) * (cc[i] - mc) for i in ix)
            alpha = min(1.0, max(0.0, covariance / variance)) if variance > 0 else 0.0
        else:
            mr, mc, alpha = statistics.mean(rr[i] for i in ix), 0.0, 0.0
        fits.append({'excluded_seed': seed, 'fit_seeds': [s for s in seeds if s != seed],
                     'alpha': alpha, 'intercept': mr - alpha * mc, 'fit_weight_sum': total})
        for i, game in enumerate(games):
            if game['seed'] == seed:
                baselines[keys[i]] = mr + alpha * (cc[i] - mc)
    return baselines, fits
