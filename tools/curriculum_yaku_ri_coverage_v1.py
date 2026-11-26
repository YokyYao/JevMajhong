"""Complete25-source yaku replay with the unchanged bounded RI sampler body."""
from types import FunctionType

from tools import curriculum_ri_coverage_replay_v2 as previous
from tools.curriculum_yaku_parent_loss_v1 import MASS

RI = previous.RI
METHOD = 'yaku25_all_originals_minimum_four_RI_sources_inverse_repeat_v1'


def build(categories, baseline, seed, batch=73, max_repeat=8, max_weight=16.):
    # Isolate only the source-mass table. Preserve the full numeric sampler,
    # feasibility solver, weights, repeat limits and original-row invariants.
    sampler = FunctionType(previous.build.__code__,
        dict(previous.build.__globals__, MASS=dict(MASS)), previous.build.__name__,
        previous.build.__defaults__, previous.build.__closure__)
    result = sampler(categories, baseline, seed, batch, max_repeat, max_weight)
    result['summary']['sampler_body_method'] = previous.METHOD
    result['summary']['method'] = METHOD
    return result
