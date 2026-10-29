"""Fit a scalar control-variate baseline outside each entire training wall stream."""
import math
import statistics


def calibrated_baselines(games, controls):
    control = {(g['seed'], g['seat']): float(g['reward']) for g in controls}
    if len(control) != len(controls):
        raise ValueError('Duplicate frozen control')
    keys = [(g['seed'], g['seat'], g['replica']) for g in games]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate candidate episode')
    seeds = sorted({g['seed'] for g in games})
    if len(seeds) < 3:
        raise ValueError('Need at least three independent training streams')
    returns = [float(g['reward']) for g in games]
    refs = [control[g['seed'], g['seat']] for g in games]
    if not all(math.isfinite(x) for x in returns + refs):
        raise ValueError('Nonfinite training return')
    fits, baselines = [], {}
    for seed in seeds:
        indices = [i for i, g in enumerate(games) if g['seed'] != seed]
        rr, cc = [returns[i] for i in indices], [refs[i] for i in indices]
        mr, mc = statistics.mean(rr), statistics.mean(cc)
        variance = statistics.pvariance(cc)
        covariance = statistics.mean((r - mr) * (c - mc) for r, c in zip(rr, cc))
        alpha = min(1.0, max(0.0, covariance / variance)) if variance > 0 else 0.0
        fits.append({'excluded_seed': seed, 'fit_seeds': [s for s in seeds if s != seed],
                     'fit_episodes': len(indices), 'alpha': alpha,
                     'mean_return': mr, 'mean_control': mc, 'intercept': mr - alpha * mc})
        for i, game in enumerate(games):
            if game['seed'] == seed:
                baselines[keys[i]] = mr + alpha * (refs[i] - mc)

    def describe(values):
        weights = [g.get('trainable_decisions', 1) for g in games]
        return {'mean': statistics.mean(values), 'std': statistics.pstdev(values),
                'rms': math.sqrt(statistics.mean(x * x for x in values)),
                'decision_weighted_rms': math.sqrt(sum(w * x * x for w, x in zip(weights, values)) / sum(weights))}

    raw = [r - c for r, c in zip(returns, refs)]
    calibrated = [r - baselines[k] for r, k in zip(returns, keys)]
    return baselines, {'method': 'leave-one-wall-stream-out affine frozen control',
                       'training_only': True, 'independent_streams': len(seeds),
                       'candidate_episodes': len(games), 'fits': fits,
                       'raw_residual': describe(raw), 'calibrated_residual': describe(calibrated)}
