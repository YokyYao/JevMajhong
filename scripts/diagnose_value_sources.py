"""Training-only grouped diagnostics; never select a native evaluation candidate."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from diagonal_value import cross_validate, ridge_fit, wall_folds
from head_rl import fingerprint
from scripts.train_point_bias import frozen


def equivalent(actual, expected):
    if isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(equivalent(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(equivalent(a, b) for a, b in zip(actual, expected))
    return actual == expected


def diagnostics(diagonal_run, bias_run):
    runs = [diagonal_run, bias_run]
    protocols, fits, saved = [], [], []
    for run in runs:
        protocol = json.loads((run / 'protocol.json').read_text())
        frozen(protocol)
        proof = json.loads((run / 'fit-audit.json').read_text())
        if proof['audit_status'] != 'passed' or proof['source_protocol_sha256'] != protocol['sha256']:
            raise ValueError('Both feature maps need matching completed fit audits')
        protocols.append(protocol)
        fits.append(json.loads((run / 'training-result.json').read_text()))
        saved.append(torch.load(run / 'private-regression.pt', map_location='cpu', weights_only=False))
    if protocols[0]['reference_head_sha256'] != protocols[1]['reference_head_sha256']:
        raise ValueError('Use the same frozen reference for both maps')
    if fits[0]['data'] != fits[1]['data']:
        raise ValueError('Feature maps must describe the same verified source data')
    if not torch.equal(saved[0]['y'], saved[1]['y']) or saved[0]['seeds'] != saved[1]['seeds']:
        raise ValueError('Paired targets or wall order differ')
    expected = json.loads((bias_run / 'fit-audit.json').read_text())['private_regression_sha256']
    if fingerprint(bias_run / 'private-regression.pt') != expected:
        raise ValueError('The audited bias features changed')
    y, seeds = saved[0]['y'].double(), saved[0]['seeds']
    maps = {'diagonal': saved[0]['x'].double(), 'bias': saved[1]['x'].double()}
    maps['joint_exploratory'] = torch.cat([maps['diagonal'], maps['bias']], dim=1)
    if any(len(x) != len(y) for x in maps.values()):
        raise ValueError('Incomplete training feature coverage')
    source_masks, sources = [], []
    for record in fits[0]['data']['sources']:
        source = ROOT / record['source']
        config = json.loads((source / 'protocol.json').read_text())['config']
        if fingerprint(source / 'rollout-stage-01.pt') != record['rollout_sha256']:
            raise ValueError('Audited source changed')
        start, length = config['training_seed_start'], config['training_seeds_per_stage']
        mask = torch.tensor([start <= s < start + length for s in seeds])
        if not mask.any():
            raise ValueError('Missing source training walls')
        source_masks.append(mask)
        sources.append({'source': record['source'], 'contrasts': int(mask.sum()),
                        'wall_streams': len({s for i, s in enumerate(seeds) if mask[i]}),
                        'nonzero_targets': int((y[mask] != 0).sum()),
                        'zero_prediction_mse': float(y[mask].square().mean())})
    if not torch.equal(torch.stack(source_masks).sum(0), torch.ones(len(y), dtype=torch.long)):
        raise ValueError('Training source membership must cover every contrast exactly once')
    alphas = protocols[1]['config']['ridge_alphas']
    folds = protocols[1]['config']['cross_validation_folds']
    reports = {}
    for name, x in maps.items():
        cv = cross_validate(x, y, seeds, alphas, folds)
        if name != 'joint_exploratory':
            index = ['diagonal', 'bias'].index(name)
            if not equivalent(cv, fits[index]['cross_validation']):
                raise ValueError(f'The audited cross-validation does not reproduce: {name}')
        prediction = torch.empty_like(y)
        for indices in wall_folds(seeds, folds):
            train = torch.ones(len(y), dtype=torch.bool)
            train[indices] = False
            prediction[indices] = x[indices] @ ridge_fit(x[train], y[train], cv['best_alpha'])
        within = [{'source': source['source'],
                   'heldout_wall_mse': float((prediction[mask] - y[mask]).square().mean()),
                   'zero_prediction_mse': source['zero_prediction_mse']}
                  for source, mask in zip(sources, source_masks)]
        transfer = []
        for source, heldout in zip(sources, source_masks):
            train = ~heldout
            inner_seeds = [s for i, s in enumerate(seeds) if train[i]]
            inner = cross_validate(x[train], y[train], inner_seeds, alphas, folds)
            coefficient = ridge_fit(x[train], y[train], inner['best_alpha'])
            mse = float((x[heldout] @ coefficient - y[heldout]).square().mean())
            transfer.append({'heldout_source': source['source'], 'selected_alpha': inner['best_alpha'],
                             'mse': mse, 'zero_prediction_mse': source['zero_prediction_mse'],
                             'alpha_selected_without_heldout_source': True})
        reports[name] = {'parameters': x.shape[1], 'grouped_cv': cv,
                         'by_source_heldout_walls': within, 'leave_source_out': transfer}
        print(json.dumps({'completed_training_diagnostic': name, 'grouped_mse': cv['best_mse'],
                          'zero_mse': cv['zero_predictor_mse']}), flush=True)
    return {'status': 'complete_training_only_diagnostic', 'source_protocols': [p['sha256'] for p in protocols],
            'feature_file_sha256': {str(r / 'private-regression.pt'): fingerprint(r / 'private-regression.pt') for r in runs},
            'sources': sources, 'reports': reports,
            'native_candidates_exported': False, 'evaluation_protocol_changed': False,
            'scope': 'Posthoc training-only diagnostic. Joint map is exploratory. Source policy and wall distributions are confounded; transfer MSE cannot establish their causal contribution or actual game-score improvement.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagonal-run', type=Path, required=True)
    parser.add_argument('--bias-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or not args.out.resolve().is_relative_to((ROOT / 'runs').resolve()):
        raise ValueError('Use a new private output path under runs/')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    result = diagnostics(args.diagonal_run, args.bias_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
