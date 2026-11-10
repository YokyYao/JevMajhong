"""Admit a frozen continuation only after both registered score comparisons."""
import json
import math
from pathlib import Path

from call_policy import digest


def checked_gain(audit, key):
    value = audit[key]
    gain = value['gain']
    bounds = value['ci95']
    if (type(gain) not in (int, float) or not math.isfinite(gain)
            or len(bounds) != 2
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in bounds)
            or not bounds[0] <= gain <= bounds[1]):
        raise ValueError('Finite complete paired statistics required')
    return gain


def choose_confirmation(development_audit, parent_audit, *, seeds):
    """Positive means advance; crossing-zero intervals are not a veto."""
    count = seeds * 4 * 3
    if (development_audit['status'] != 'passed_complete'
            or development_audit['stage'] != 'development'
            or development_audit['games'] != 2 * count
            or development_audit['independent_wall_streams'] != seeds):
        raise ValueError('Complete registered development audit required')
    baseline_gain = checked_gain(development_audit, 'overall_paired_gain')
    if baseline_gain <= 0:
        return {'decision': 'skip_nonpositive_oldB', 'oldB_gain': baseline_gain}
    if parent_audit is None:
        raise ValueError('Positive oldB signal still requires completed parent comparison')
    if (parent_audit['status'] != 'passed_complete'
            or parent_audit['games'] != 2 * count
            or parent_audit['new_parent_games'] != count
            or parent_audit['reused_candidate_games'] != count
            or any(parent_audit.get(k) is not True for k in (
                'candidate_games_source_verified', 'source_identity_passed',
                'native_parity_passed', 'summary_recomputed'))):
        raise ValueError('Complete audited matched-parent comparison required')
    parent_gain = checked_gain(parent_audit, 'overall_parent_paired_gain')
    if parent_audit['overall_parent_paired_gain']['independent_wall_streams'] != seeds:
        raise ValueError('Parent wall count changed')
    return {'decision': 'confirm' if parent_gain > 0 else 'skip_nonpositive_parent',
            'oldB_gain': baseline_gain, 'parent_gain': parent_gain}


def confirmation_decision(root, config):
    """Check original frozen files before allowing the existing GPU evaluator."""
    root = Path(root)
    if config['stage'] != 'independent_confirmation':
        raise ValueError('Independent confirmation config required')
    source = json.loads((root / config['source_config']).read_text())
    if (config['seed_start'] != source['confirmation_seed_start']
            or config['seeds'] != source['confirmation_seeds']):
        raise ValueError('Confirmation walls changed')
    development = root / config['development_out']
    protocol = json.loads((development / 'protocol.json').read_text())
    dev_cfg = protocol['config']
    if (dev_cfg['source_config'] != config['source_config']
            or dev_cfg['out'] != config['development_out']
            or dev_cfg['stage'] != 'development'
            or dev_cfg['seed_start'] != source['development_seed_start']
            or dev_cfg['seeds'] != source['development_seeds']
            or config['out'] == config['development_out']):
        raise ValueError('Development source or wall registration differs')
    # This verifies preparation, candidate/reference/parent identities, source
    # hashes, and the completed development raw records without loading a model.
    from scripts.evaluate_feature_parent import registration
    expected_parent_protocol, _, _ = registration(dev_cfg)
    dev_audit = json.loads((development / 'evaluation-audit.json').read_text())
    if checked_gain(dev_audit, 'overall_paired_gain') <= 0:
        return choose_confirmation(dev_audit, None, seeds=dev_cfg['seeds'])
    parent = root / dev_cfg['paired_parent_out']
    parent_audit = json.loads((parent / 'evaluation-audit.json').read_text())
    if json.loads((parent / 'protocol.json').read_text()) != expected_parent_protocol:
        raise ValueError('Matched-parent source or weights changed')
    if digest(parent / 'games-private.jsonl') != parent_audit['raw_parent_games_sha256']:
        raise ValueError('Audited parent records changed')
    summary = json.loads((parent / 'summary.json').read_text())
    if summary['overall']['candidate_minus_reference'] != parent_audit['overall_parent_paired_gain']:
        raise ValueError('Parent summary differs from completed audit')
    return choose_confirmation(dev_audit, parent_audit, seeds=dev_cfg['seeds'])
