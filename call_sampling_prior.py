"""Fixed inverse-inclusion correction for a labelled TRAIN sampling experiment.

This is a diagnostic under a calibrated sampled-posterior assumption. It does
not establish strength, and does not change the deployed policy.
"""
import math

CLASSES = ('closed_chi', 'closed_pon', 'open_chi', 'open_pon', 'none')


def inverse_inclusion(source_counts, sampled_counts):
    if set(source_counts) != set(CLASSES) or set(sampled_counts) != set(CLASSES):
        raise ValueError('Every reaction class must be registered')
    if any(type(source_counts[k]) is not int or type(sampled_counts[k]) is not int
           or not 0 < sampled_counts[k] <= source_counts[k] for k in CLASSES):
        raise ValueError('Positive supported inclusion counts required')
    return {key: source_counts[key] / sampled_counts[key] for key in CLASSES}


def option_class(option):
    if option['type'] == 'none':
        return 'none'
    if option['type'] not in ('chi', 'pon'):
        raise ValueError('Only chi/pon/pass are in this diagnostic')
    closed = option.get('call_features', {}).get('loses_closed_status')
    if type(closed) is not bool:
        raise ValueError('Known public closed status required')
    return ('closed_' if closed else 'open_') + option['type']


def corrected_probabilities(criteria, probabilities, weights):
    # Only public options and model outputs enter this calculation. No expert
    # label, validation outcome, score or adjustable threshold is accepted.
    if list(criteria) != list(probabilities) or not criteria:
        raise ValueError('Exact original option keys/order required')
    if set(weights) != set(CLASSES) or any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError('Finite positive fixed weights required')
    if any(not math.isfinite(v) or v < 0 for v in probabilities.values()) or not math.isclose(math.fsum(probabilities.values()), 1., abs_tol=1e-5):
        raise ValueError('Normalized input probabilities required')
    weighted = {key: probabilities[key] * weights[option_class(option)]
                for key, option in criteria.items()}
    total = math.fsum(weighted.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError('Invalid corrected probability mass')
    return {key: value / total for key, value in weighted.items()}
