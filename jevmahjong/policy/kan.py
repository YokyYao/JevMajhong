"""Apply a specialized head only to public legal kan windows."""
from .routing import HeadPolicyClient, HybridPolicy

KAN_TYPES = frozenset(('ankan', 'daiminkan', 'kakan'))


def is_kan_request(request):
    return any(c['type'] in KAN_TYPES for c in request['questions']['action']['criteria'].values())


class KanPolicyClient:
    def __init__(self, ordinary, calling, heads):
        self.frozen = HybridPolicy(ordinary, calling)
        self.updated = HeadPolicyClient(ordinary, calling, heads)

    def system_one(self, **request):
        client = self.updated if is_kan_request(request) else self.frozen
        return client.system_one(**request)
