"""Select a discard after an actual reach action, before reach acceptance.

RiichiEnv0.4.11 sets riichi_declared after the declaration discard, so that
acceptance flag must not gate the immediately following discard observation.
"""
from tools.curriculum_risk_teacher import guard_diagnostic


def category(types, state, previous_reach=False, existing=False):
    if existing or types != {'dahai'} or not previous_reach:
        return None
    return 'post_reach_discard'
