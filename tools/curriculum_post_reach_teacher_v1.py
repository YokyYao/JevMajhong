"""Select the actual immediate post-reach discard, never a later locked draw."""
from tools.curriculum_risk_teacher import guard_diagnostic


def category(types, state, previous_reach=False, existing=False):
    if existing or types != {'dahai'} or not previous_reach or state['seat'] not in state['riichi']:
        return None
    return 'post_reach_discard'
