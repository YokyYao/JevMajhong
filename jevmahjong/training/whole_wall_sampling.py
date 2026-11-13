"""Limit correlated decision windows without inspecting rollout outcomes."""
from collections import Counter


def balanced_subset(candidates, routes, requested, budget, wall_cap):
    key=lambda row:tuple(row['episode_key'])
    order={key(row):index for index,row in enumerate(candidates)}
    chosen={};counts=Counter()

    def take(pool,number):
        while sum(key(row) in chosen for row in pool)<number:
            available=[row for row in pool if key(row) not in chosen
                and counts[row['episode_key'][0]]<wall_cap]
            if not available:raise ValueError('Insufficient whole-wall-balanced public coverage')
            row=min(available,key=lambda r:(counts[r['episode_key'][0]],order[key(r)]))
            chosen[key(row)]=row;counts[row['episode_key'][0]]+=1

    for name,number in sorted(requested.items()):
        take([group['row'] for group in routes.get(name,[])],number)
    take(candidates,budget)
    if len(chosen)!=budget:raise ValueError('Public route reserves exceed distinct-state budget')
    return list(chosen.values())
