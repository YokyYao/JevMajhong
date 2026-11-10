"""Reuse complete independent candidate games for a matched frozen-parent check.

The original parent evaluator remains unchanged for its frozen development
protocol. This entry point adds a separate output path after verifying the
unmodified independent-confirmation config and all original source identities.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from scripts import evaluate_feature_parent as paired

_original_registration = paired.registration


def confirmation_registration(config):
    if config['stage'] != 'independent_confirmation':
        raise ValueError('Independent confirmation required for this comparison')
    original = {k: v for k, v in config.items() if k != 'paired_parent_out'}
    if config['paired_parent_out'] == config['out']:
        raise ValueError('Independent parent comparison needs a separate output')
    protocol, source, parent_cfg = _original_registration(original)
    if (config['seed_start'] != source['confirmation_seed_start']
            or config['seeds'] != source['confirmation_seeds']):
        raise ValueError('Independent confirmation walls changed')
    # Retain every original identity and the original parent evaluator hash.
    # Only this additional entry point and its separate output are added.
    protocol['config'] = dict(config)
    protocol['source_confirmation_config'] = original
    protocol['code_sha256']['scripts/evaluate_feature_confirmation_parent.py'] = digest(
        ROOT / 'scripts/evaluate_feature_confirmation_parent.py')
    protocol['comparison'] = ('Continued candidate minus exact frozen feature parent '
        'on independent confirmation walls; candidate games reused exactly')
    return protocol, source, parent_cfg


def main():
    # The shared main/auditor resolve registration in their own module. Adapt
    # just that hook; all game generation, reuse, parity and score checks remain.
    previous = paired.registration
    paired.registration = confirmation_registration
    try:
        paired.main()
    finally:
        paired.registration = previous


if __name__ == '__main__':
    main()
