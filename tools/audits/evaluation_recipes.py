"""Check fresh preflight/development/confirmation links before registration."""
import argparse
import json
import math
from pathlib import Path


IDENTITY_FIELDS = (
    'discard_checkpoint', 'call_checkpoint', 'call_context_baseline', 'candidate_run',
    'updated_scopes', 'candidate_activation', 'train_scopes', 'decision_filter',
    'fast_gamma', 'confirmation_objective', 'max_dealin_rate_increase',
    'required_fit_audit_config', 'required_call_head_audit_config',
    'parity_source_run', 'candidate_action_admission',
)


def check_recipes(preflight, development, confirmation, workspace='.'):
    """Check recipes only; weights, fit eligibility and strength need real audits."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(preflight.get('preflight_only') is True, 'Explicit native preflight required')
    require(development.get('preflight_only', False) is False
            and confirmation.get('preflight_only', False) is False,
            'Match stages cannot be preflight-only')
    require(preflight.get('stage') == development.get('stage') == 'development'
            and confirmation.get('stage') == 'independent_confirmation', 'Stage identities differ')
    for key in IDENTITY_FIELDS:
        require(preflight.get(key) == development.get(key) == confirmation.get(key),
                'Comparison identity differs: ' + key)
    require(development.get('confirmation_objective') == 'hand_win', 'Hand-win objective required')
    require(development.get('updated_scopes') in (['calling'], ['ordinary']), 'One updated scope required')
    audit_keys = [key for key in ('required_fit_audit_config', 'required_call_head_audit_config')
                  if development.get(key)]
    require(len(audit_keys) == 1, 'Exactly one prescribed fit audit required')
    for key in ('discard_checkpoint', 'call_checkpoint', 'candidate_run', 'out', audit_keys[0]):
        require(isinstance(development.get(key), str) and bool(development[key]),
                'Nonempty recipe path required: ' + key)
    risk = development.get('max_dealin_rate_increase')
    gamma = development.get('fast_gamma')
    require(type(risk) in (int, float) and math.isfinite(risk) and 0 <= risk <= 1,
            'Finite deal-in limit required')
    require(type(gamma) in (int, float) and math.isfinite(gamma) and 0 < gamma <= 1,
            'Finite speed discount required')
    ranges = []
    for cfg in (preflight, development, confirmation):
        require(type(cfg.get('seed_start')) is int and type(cfg.get('seeds')) is int
                and cfg['seeds'] > 0, 'Explicit positive wall counts required')
        require(not cfg.get('reference_cache_out'), 'Fresh-reference recipes cannot reuse matches')
        ranges.append((cfg['seed_start'], cfg['seed_start'] + cfg['seeds']))
    require(ranges[0] == ranges[1], 'Preflight must check the declared development recipe')
    require(max(ranges[1][0], ranges[2][0]) >= min(ranges[1][1], ranges[2][1]),
            'Development and confirmation walls overlap')
    resolve = lambda p: (Path(workspace) / p).resolve()
    require(all(isinstance(c.get('out'), str) and c['out'] for c in
                (preflight, development, confirmation)), 'Distinct output paths required')
    require(len({resolve(c['out']) for c in (preflight, development, confirmation)}) == 3,
            'Evaluation output paths overlap')
    require(isinstance(confirmation.get('development_out'), str)
            and resolve(confirmation['development_out']) == resolve(development['out']),
            'Confirmation points to a different development run')
    return {'status': 'passed_recipe_links', 'candidate_run': development['candidate_run'],
            'development_out': development['out'], 'confirmation_out': confirmation['out'],
            'development_wall_range': ranges[1], 'confirmation_wall_range': ranges[2],
            'actual_weights_or_fit_eligibility_verified': False,
            'independent_strength_proven': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('preflight', 'development', 'confirmation'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--workspace', default='.')
    args = parser.parse_args()
    configs = [json.loads(getattr(args, name).read_text())
               for name in ('preflight', 'development', 'confirmation')]
    print(json.dumps(check_recipes(*configs, workspace=args.workspace)))


if __name__ == '__main__':
    main()
