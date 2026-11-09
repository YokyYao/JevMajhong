"""Separate, supervised chi/pon readout; no changes to the frozen discard head."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

CALL_TYPES = {'chi', 'pon', 'none'}
METADATA = ('scores', 'dealer', 'round_wind', 'honba', 'sticks', 'hand_number', 'riichi')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def is_call_request(request):
    criteria = request['questions']['action']['criteria']
    kinds = {v['type'] for v in criteria.values()}
    return len(criteria) >= 2 and 'none' in kinds and bool(kinds & {'chi', 'pon'}) and kinds <= CALL_TYPES


def request_hash(request):
    value = copy.deepcopy(request)
    value.pop('_meta', None)
    value['questions']['action'].pop('label', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def screenshot_view(record, unknown=False):
    """Remove information unavailable in one screenshot; never invent tile IDs."""
    result = copy.deepcopy(record)
    state = result['state']
    criteria = result['questions']['action']['criteria']
    call = next(v for v in criteria.values() if v['type'] in {'chi', 'pon'})
    from desktop.jev_sidecar.tiles import HONORS
    tile = HONORS.get(call['pai'], '0' + call['pai'][1] if call['pai'].endswith('r') else call['pai'])
    source = record['_meta']['reaction_relative_seat']
    for field in ('tsumogiri_flags', 'riichi_sutehais', 'last_tedashis', 'last_discard'):
        state[field] = None
    state['screen_observation'] = {'temporal_information': 'unknown',
        'metadata_confirmed': not unknown, 'reaction': {'tile': tile, 'seat': source,
            'offered': sorted({v['type'] for v in criteria.values()} - {'none'})}}
    if unknown:
        for field in METADATA:
            state[field] = None
    return result


def metrics(rows):
    import math
    matrix = {a: Counter() for a in sorted(CALL_TYPES)}
    nll, correct, selected_calls, exact_calls, predicted_calls = 0., 0, 0, 0, 0
    for row in rows:
        gold, pred = row['gold'], row['pred']
        matrix[gold][pred] += 1
        correct += row['exact']
        selected_calls += gold in {'chi', 'pon'}
        predicted_calls += pred in {'chi', 'pon'}
        exact_calls += row['exact'] and gold in {'chi', 'pon'}
        nll -= math.log(max(row['target_probability'], 1e-30))
    classes = {}
    for kind in sorted(CALL_TYPES):
        tp, support = matrix[kind][kind], sum(matrix[kind].values())
        predicted = sum(v[kind] for v in matrix.values())
        precision, recall = tp / max(1, predicted), tp / max(1, support)
        classes[kind] = {'n': support, 'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / max(1e-30, precision + recall)}
    return {'n': len(rows), 'exact_accuracy': correct / max(1, len(rows)),
        'nll': nll / max(1, len(rows)), 'macro_f1': sum(v['f1'] for v in classes.values()) / 3,
        'by_target': classes, 'predicted_calls': predicted_calls,
        'exact_call_recall': exact_calls / max(1, selected_calls),
        'note': 'Expert imitation metrics, not strategic optimality or win rate.'}


class CallRouter:
    """Serialize model access and restore its original head even after failure."""
    def __init__(self, client, call_head):
        from threading import RLock
        self.client, self.call_head = client, call_head
        self.lock = RLock()

    def system_one(self, **request):
        with self.lock:
            model = self.client.runtime.model
            original = model.head
            use_call = is_call_request(request)
            if use_call:
                request = copy.deepcopy(request)
                screen = request['state'].get('screen_observation', {})
                if screen and not screen.get('metadata_confirmed', False):
                    for field in METADATA:
                        request['state'][field] = None
                model.head = self.call_head
            try:
                return self.client.system_one(**request)
            finally:
                model.head = original
