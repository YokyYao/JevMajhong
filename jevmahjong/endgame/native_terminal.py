"""Admit verified native terminal values without duplicating old model inputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .collect import digest
from .context import placement_objective, target_from_utilities, TRAILING_INPUT_VERSION
from .native_retention import native_record

NATIVE_TERMINAL_TAG = 'ordinary_fourth16_v1'


def is_native_terminal(meta):
    return meta.get('native_terminal_supervision') == NATIVE_TERMINAL_TAG


def terminal_native_record(row, label, file_sha256, config):
    """The same source ID/input receives genuine terminal targets, never an expert label."""
    from tools.label_native_endgame import validate_label
    validate_label(label, row, {'target_temperature': .1}, config['identity_sha256'])
    if row['rank'] != 4:
        raise ValueError('Only original current-fourth ordinary frames have terminal supervision')
    record = native_record(row)
    q, meta = record['questions']['action'], record['_meta']
    keys = list(q['criteria'])
    values = {k: sum(placement_objective(o['ranks'][row['seat']]) for o in outcomes) / 16
              for k, outcomes in zip(keys, label['outcomes'])}
    escape = {k: sum(o['ranks'][row['seat']] < 4 for o in outcomes) / 16
              for k, outcomes in zip(keys, label['outcomes'])}
    probability = target_from_utilities(list(values.values()), config['target_temperature'])
    q.update(src='endgame_terminal', target=dict(zip(keys, probability)))
    q['label'] = max(q['target'], key=q['target'].get)
    meta.pop('native_reference_pending')
    meta.update(source='native_fourth_terminal', bucket='fourth',
        label_type='complete_native_terminal_MC', native_terminal_supervision=NATIVE_TERMINAL_TAG,
        native_retention_scope='trailing', expected_terminal_utilities=values,
        terminal_objective='top2_primary_v1', terminal_escape4_probabilities=escape,
        target_basis='complete_native_terminal_MC_before_frozen_reference_tilt',
        terminal_label_provenance={'set': 'native_fourth16', 'worlds_per_action': 16,
            'source_target_temperature': .1, 'file_sha256': file_sha256})
    if record['state'] != row['student_request']['state'] or q['criteria'] != row['student_request']['questions']['action']['criteria']:
        raise ValueError('Terminal supervision must not change the student input')
    return record


def load_terminal_native_records(config, native_rows):
    root = config.get('native_terminal_label_root')
    if root is None:
        return {}
    if (config.get('input_version') != TRAILING_INPUT_VERSION
            or config.get('reference_mode') != 'released_original_policy'
            or config.get('fourth_escape_constraint') is not True
            or config.get('native_terminal_worlds') != 16
            or config.get('enforce_native_terminal_escape_guard') is not True
            or config.get('training_objective') != 'top2_primary_v1'):
        raise ValueError('Native terminal admission requires the registered original-policy and escape objective')
    root = Path(root)
    manifest = json.loads((root / 'selected-positions-private.json').read_text())
    receipt = json.loads((root / 'label-verification-private.json').read_text())
    protocol = json.loads((root / 'protocol-private.json').read_text())
    from .futures import bundle_identity
    identity = bundle_identity(config['checkpoint'])['identity_sha256']
    expected = [r for r in native_rows if r['rank'] == 4]
    if (manifest != expected or len(manifest) != 40 or len({r['seed'] for r in manifest}) != 40
            or Counter(r['split'] for r in manifest) != {'train': 32, 'validation': 8}
            or digest(manifest) != config['native_terminal_positions_sha256']
            or receipt['status'] != 'all40_native_fourth_terminal_menus_verified'
            or receipt['positions'] != 40 or receipt['future_worlds'] != 16
            or receipt['protocol_sha256'] != digest(protocol)
            or receipt['positions_sha256'] != digest(manifest)
            or protocol['positions_sha256'] != digest(manifest)
            or protocol['baseline_identity']['identity_sha256'] != identity
            or protocol['configuration']['continuation_policy'] != 'released_original_v1'
            or protocol['configuration']['future_worlds'] != 16
            or receipt['continuation_policy'] != 'released_original_v1'
            or receipt['terminal_rollouts'] != protocol['terminal_rollout_budget']
            or set(receipt['labels_sha256']) != {r['id'] for r in manifest}):
        raise ValueError('Original native terminal source/split/identity/final verification differs')
    records = {}
    for row in manifest:
        path = root / 'labels' / (row['id'].split(':', 1)[1] + '.json')
        data = path.read_bytes(); sha256 = hashlib.sha256(data).hexdigest()
        if sha256 != receipt['labels_sha256'][row['id']]:
            raise ValueError('Original native terminal label changed after final verification')
        records[row['id']] = terminal_native_record(row, json.loads(data), sha256,
            dict(config, identity_sha256=identity))
    return records
