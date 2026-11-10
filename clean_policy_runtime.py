"""Serving adapter for clean full-policy models and untrained rare-action fallback."""
from collections import Counter
import copy

from clean_policy_data import policy_request


class CleanPolicyRouter:
    def __init__(self, ordinary, candidate):
        self.ordinary, self.candidate = ordinary, candidate
        self.counts = Counter()

    def system_one(self, **request):
        from jevany.data import api_request
        kinds = {v['type'] for v in request['questions']['action']['criteria'].values()}
        learned = (bool(kinds) and kinds <= {'dahai', 'reach'} and 'dahai' in kinds) or (
            bool(kinds & {'chi', 'pon'}) and kinds <= {'chi', 'pon', 'none'})
        if learned:
            self.counts['learned'] += 1
            screen = request['state'].get('screen_observation', {})
            if screen and not screen.get('metadata_confirmed', False):
                request = copy.deepcopy(request)
                confirmed = set(screen.get('confirmed_metadata', []))
                for field in ('scores','dealer','round_wind','honba','sticks','hand_number','riichi'):
                    if field not in confirmed:
                        request['state'][field] = None
            return self.candidate.system_one(**api_request(policy_request(request)))
        self.counts['rare_action_fallback'] += 1
        return self.ordinary.system_one(**api_request(request))
