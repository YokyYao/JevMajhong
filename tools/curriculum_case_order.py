"""Deterministic seat order for simultaneous decisions and private traces."""


def ordered_seats(observations):
    if any(type(p) is not int or p not in range(4) for p in observations):
        raise ValueError('Native observing seats must be integers0..3')
    return sorted(observations.items())


def canonical_trace(trace):
    """Keep batch/time order; only simultaneous actors/consumed sets commute."""
    result=[]
    for batch in trace:
        if len({a['player'] for a in batch})!=len(batch):raise ValueError('Duplicate actor within a joint action')
        result.append([dict(player=a['player'],type=a['type'],tile=a['tile'],consumed=sorted(a['consumed']))
            for a in sorted(batch,key=lambda a:a['player'])])
    return result
