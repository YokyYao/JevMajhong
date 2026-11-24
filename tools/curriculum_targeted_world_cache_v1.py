"""Durable complete-world cache; a world contains every current legal arm.

Caller must validate GPU allocation, frozen sources and prior writer exits.
This component never samples a replacement world or computes a partial target.
"""
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from tools.curriculum_transfer import sha

METHOD = 'complete_current_menu_world_cache_exact_context_and_diagnostics_v1'


def normalized(value):
    # First convert JSON-compatible integer keys/tuples exactly as case files
    # do; only then sort keys, so mixed Python key types are unambiguous.
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


def digest(value):
    return hashlib.sha256(json.dumps(normalized(value), sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def slot(context):
    return dict(source_id=context['source']['id'], block=context['block'], world=context['world'])


def context(identity, source, block, world, allocation, receipt, planned, planner_before, seat, anchor):
    if (type(block) is not int or block not in range(2) or type(world) is not int or world not in range(16)
            or type(seat) is not int or seat not in range(4) or len(anchor) != 4
            or not 2 <= len(planned['candidates']) <= 16):
        raise ValueError('Fixed paired16-world indices and complete2..16-arm menu required')
    return normalized(dict(identity=identity, source=source, block=block, world=world,
        allocation=allocation, receipt=receipt, public_seed=planned['public_seed'],
        baseline=planned['baseline'], candidates=list(planned['candidates']),
        planner_before=dict(planner_before), seat=seat, anchor_scores=list(anchor)))


def validate_payload(packet, validator):
    if packet['method'] != METHOD or packet['context_sha256'] != digest(packet['context']):
        raise ValueError('Exact complete-world cache context required')
    c = packet['context']; keys = c['candidates']; paired = packet['paired']; plans = packet['plans']
    if (len(keys) != len(set(keys)) or not 2 <= len(keys) <= 16
            or set(paired) != set(keys) or set(plans) != set(keys)
            or c['baseline'] not in keys):
        raise ValueError('Every original current legal arm and future-plan log required')
    before = c['planner_before']; after = packet['planner_after']
    allowed = {'legal_win_offers', 'outside_late_scope', 'unsupported_current_kan_offer',
        'budget_fallback_wins', 'planned_calls', 'inner_terminal_arms', 'selected_declines'}
    if (not set(before) <= set(after)
            or not set(after) <= allowed
            or any(type(v) is not int or v < 0 for v in (*before.values(), *after.values()))
            or any(after[k] < v for k, v in before.items())):
        raise ValueError('Exact nondecreasing observer diagnostic state required')
    planned = sum(len(v) for v in plans.values())
    inner = 0; declines = 0; offers = 0
    for key in keys:
        result = paired[key]
        if (result['full_match_completed'] is not True
                or result['opponents_no_legal_win_declines_verified'] is not True
                or type(result['rank']) is not int or result['rank'] not in (1, 2, 3, 4)
                or result['top2'] != (result['rank'] <= 2)
                or result['net_from_anchor'] != result['terminal_scores'][c['seat']]-c['anchor_scores'][c['seat']]
                or type(result['observer_hook_calls']) is not int or result['observer_hook_calls'] < 0):
            raise ValueError('Complete actual match, observer and terminal-score accounting required')
        checked = validator(result['merged_mjai_log'], result['terminal_scores'])
        if checked['hands'] != result['merged_ledger_hands']:
            raise ValueError('Cached native ledger hand count differs')
        offers += result['observer_hook_calls']
        if not isinstance(plans[key], list) or len(plans[key]) > 4:
            raise ValueError('Original bounded per-arm future observer plans required')
        for plan in plans[key]:
            if (plan['full_terminal_arms'] != 8
                    or plan['decision']['selected'] not in (plan['accept'], plan['decline'])
                    or plan['accept'] == plan['decline']):
                raise ValueError('Original two-arm/four-world future-plan contract required')
            inner += plan['full_terminal_arms']; declines += plan['decision']['selected'] == plan['decline']
    for key, expected in (('planned_calls', planned), ('inner_terminal_arms', inner),
            ('selected_declines', declines), ('legal_win_offers', offers)):
        if after.get(key, 0)-before.get(key, 0) != expected:
            raise ValueError('Cached per-world observer diagnostic differs: '+key)
    fallback = sum(after.get(k, 0)-before.get(k, 0) for k in
        ('outside_late_scope', 'unsupported_current_kan_offer', 'budget_fallback_wins'))
    if planned+fallback != offers: raise ValueError('Every observer win offer must be accounted for')
    return packet


class CompleteWorldCache:
    def __init__(self, directory, previous, identity, validator):
        self.directory = Path(directory); self.previous = [Path(p) for p in previous]
        self.identity = normalized(identity); self.validator = validator
        self.used = {}; self.created = {}; self.stats = Counter()

    def make_context(self, source, block, world, allocation, receipt, planned, planner_before, seat, anchor):
        return context(self.identity, source, block, world, allocation, receipt, planned, planner_before, seat, anchor)

    def _paths(self, c):
        name = digest(slot(c))+'.json'
        return [root/name for root in (*self.previous, self.directory) if (root/name).exists()]

    def lookup(self, c):
        if c['identity'] != self.identity: raise ValueError('Wrong immutable collection/cache identity')
        packets = []
        for path in self._paths(c):
            with path.open() as stream: packet = json.load(stream)
            if normalized(packet['context']) != normalized(c):
                raise ValueError('The same fixed world slot has a changed source/allocation/diagnostic context')
            validate_payload(packet, self.validator)
            packets.append(packet); self.used[str(path.resolve())] = sha(path)
        if not packets:
            self.stats['misses'] += 1
            return None
        if any(packet != packets[0] for packet in packets[1:]):
            raise ValueError('Never choose between conflicting completed-world results')
        self.stats['reused_worlds'] += 1
        return copy.deepcopy(packets[0])

    def publish(self, c, paired, plans, planner_after):
        if c['identity'] != self.identity: raise ValueError('Wrong immutable collection/cache identity')
        packet = normalized(dict(method=METHOD, context=c, context_sha256=digest(c),
            paired=paired, plans=plans, planner_after=dict(planner_after)))
        validate_payload(packet, self.validator)
        if self._paths(c): raise ValueError('Never replace an existing completed-world slot')
        self.directory.mkdir(exist_ok=True)
        target = self.directory/(digest(slot(c))+'.json')
        with tempfile.NamedTemporaryFile(mode='w', dir=self.directory,
                prefix=target.name+'.partial-', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(packet, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            stream.flush(); os.fsync(stream.fileno())
        # Atomic exclusive publication; an interrupted temporary is never a hit.
        os.link(temporary, target); temporary.unlink()
        self.created[str(target.resolve())] = sha(target); self.stats['published_worlds'] += 1
        return packet

    def seal(self, path):
        from tools.curriculum_calls_post_reach_data_v1 import write
        files = {**self.used, **self.created}
        if any(sha(p) != h for p, h in files.items()): raise ValueError('A consumed complete-world record changed')
        write(path, dict(method=METHOD, files=files, statistics=dict(self.stats),
            original_collection_identity=self.identity, partial_world_targets_admitted=False,
            all_current_arms_required=True, teacher_optimality_not_proven=True))
        return Path(path)
