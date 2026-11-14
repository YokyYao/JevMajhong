"""Apply a specialized head only to public legal kan windows."""
from .routing import HeadPolicyClient, HybridPolicy

KAN_TYPES = frozenset(('ankan', 'daiminkan', 'kakan'))


def is_kan_request(request):
    return any(c['type'] in KAN_TYPES for c in request['questions']['action']['criteria'].values())


def request_eligible(request):
    """Reproduce PolicyAgent's discard guard from the public kan request."""
    if not is_kan_request(request):
        raise ValueError('Legal kan window required')
    criteria = list(request['questions']['action']['criteria'].values())
    all_indices = list(range(len(criteria)))
    state = request['state']; threats = set(state['riichi']) - {state['seat']}
    if any(c['type'] in ('hora', 'ron', 'tsumo') for c in criteria) or (
            not threats and any(c['type'] == 'reach' for c in criteria)):
        return all_indices
    discards = [(i, c) for i, c in enumerate(criteria) if c['type'] == 'dahai']
    if not discards:
        return all_indices
    fold = bool(threats) and min(c['sh'] for _, c in discards) >= 1
    _, ref = min(discards, key=lambda item: (
        len(threats - set(item[1]['safe'])) if fold else 0,
        item[1]['sh'], -item[1]['ukeire'], item[1]['red'], item[0]))
    risk = len(threats - set(ref['safe']))
    return [i for i, c in enumerate(criteria) if c['type'] != 'dahai' or (
        c['sh'] <= ref['sh'] and (not fold or len(threats - set(c['safe'])) <= risk))]


class KanPolicyClient:
    def __init__(self, ordinary, calling, heads):
        self.frozen = HybridPolicy(ordinary, calling)
        self.updated = HeadPolicyClient(ordinary, calling, heads)

    def system_one(self, **request):
        client = self.updated if is_kan_request(request) else self.frozen
        return client.system_one(**request)
