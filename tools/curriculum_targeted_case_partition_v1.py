"""Partition fixed missing cases without changing any case/world calculation.

This contains no launcher and cannot admit a partial shard into training.
The collector's RNG seeds are fixed within each case, independent of its
position. Local output indices change; original source IDs and order remain.
"""
import ast
import copy

from tools import curriculum_targeted_win_resume_v2 as original
from tools import curriculum_targeted_win_resume_v3 as cached


def validate_specs(specs):
    if not isinstance(specs,list) or not specs:
        raise ValueError('The complete original nonempty case selection is required')
    ids = [spec['id'] for spec in specs]
    if any(not isinstance(rid,str) or not rid for rid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Distinct original source IDs required')
    return ids


def partition_missing(specs,completed_ids,lanes=3):
    """Round-robin only by original position, never by outcomes or confidence."""
    ids = validate_specs(specs)
    if type(lanes) is not int or not 1 <= lanes <= 3:
        raise ValueError('Explicit one-to-three case workers required')
    completed = set(completed_ids)
    if not completed <= set(ids):
        raise ValueError('A completed case is outside the fixed original population')
    missing = [rid for rid in ids if rid not in completed]
    return [missing[index::min(lanes,len(missing))]
        for index in range(min(lanes,len(missing)))] if missing else []


def select_specs(specs,assigned_ids):
    ids = validate_specs(specs)
    if (not isinstance(assigned_ids,list) or not assigned_ids
            or len(set(assigned_ids)) != len(assigned_ids) or not set(assigned_ids) <= set(ids)):
        raise ValueError('An explicit distinct nonempty subset of the fixed cases is required')
    # Preserve original source order even when a caller supplies IDs out of order.
    selected = [spec for spec in specs if spec['id'] in set(assigned_ids)]
    if len(selected) != len(assigned_ids): raise ValueError('Assigned case membership differs')
    return selected


def collector_tree(source):
    """Add one pre-loop selection to the unchanged complete-world collector."""
    tree = cached.collector_tree(source)
    function, loop = original.original_loop(tree)
    body_before = ast.dump(ast.Module(body=copy.deepcopy(loop.body),type_ignores=[]))
    changes = 0
    for node in function.body:
        if (isinstance(node,ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0],ast.Name) and node.targets[0].id == 'specs'):
            expected = "reg['selected'][shard * c['views_per_shard']:(shard + 1) * c['views_per_shard']]"
            if ast.unparse(node.value) != expected:
                raise ValueError('Review the sole original fixed shard selection expression')
            node.value = ast.Call(func=ast.Name(id='select_specs',ctx=ast.Load()),
                args=[node.value,ast.Name(id='ASSIGNED_CASE_IDS',ctx=ast.Load())],keywords=[])
            changes += 1
    if changes != 1:
        raise ValueError('Exactly one explicit case-selection adaptation required')
    if ast.dump(ast.Module(body=loop.body,type_ignores=[])) != body_before:
        raise ValueError('Case sampling, planner, teacher, scoring, cache or labels changed')
    return ast.fix_missing_locations(tree)
