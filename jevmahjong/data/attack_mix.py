"""Append audited attacking futures to an admitted raw training mixture."""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import torch

from jevmahjong.core.io import digest, write
from jevmahjong.policy.ordinary import is_attack_request, request_eligible
from jevmahjong.training.utility import hand_utility
from .base_lora_pool import phase, position
from .bootstrap_targets import bootstrap_probabilities
from .extend_pool import audited_records, canonical_groups
from .relabel_targets import bind

POLICY = 'append_audited_attacking_bootstrap_targets_v1'
PHASES = {'closed_discard', 'open_discard'}


def bootstrap_target(rows, config, seed, temperature=.05, replicates=256):
    """Resample all alternatives together; serialize only public input and target."""
    if not rows: raise ValueError('A complete nonempty alternative menu is required')
    first = rows[0]; request = first['public_request']; eligible = first['eligible']
    if (not is_attack_request(request) or eligible != request_eligible(request)
            or first['teacher_action'] not in eligible):
        raise ValueError('Actual non-threat serving candidates required')
    teacher = first['teacher_action']; alternatives = [r['action'] for r in rows]
    if len(set(alternatives)) != len(rows) or set(alternatives) != set(eligible) - {teacher}:
        raise ValueError('Every served alternative must have exactly one paired future menu')
    worlds = first['counterfactual_worlds']; count = len(worlds)
    if count != config['worlds'] or count < 2:
        raise ValueError('Complete registered future count required')
    values = np.zeros((count, len(eligible), 4), dtype=np.float64)
    for row in rows:
        if (row['public_request'] != request or row['eligible'] != eligible
                or row['teacher_action'] != teacher or row['episode_key'][:3] != first['episode_key'][:3]
                or row['hand_index'] != first['hand_index'] or row['draws_before'] != first['draws_before']
                or len(row['counterfactual_worlds']) != count):
            raise ValueError('Alternatives differ from their common public origin')
        for index, world in enumerate(row['counterfactual_worlds']):
            parent = worlds[index]
            if any(world[k] != parent[k] for k in ('future_seed', 'future_wall_sha256', 'control')):
                raise ValueError('Alternatives must share ordered futures and control outcomes')
            arms = {name: hand_utility(world[name]['mjai_log'], first['hand_index'],
                first['episode_key'][1], first['draws_before'], config['fast_gamma'], config['reward'])
                for name in ('control', 'branch')}
            values[index, eligible.index(row['action'])] = [
                arms['branch'][key] - arms['control'][key]
                for key in ('utility', 'own_win', 'fast_reward', 'dealin')]
    group = 'native:' + str(first['episode_key'][0])
    identity = position(request, group)
    label_seed = int(hashlib.sha256((str(seed) + '|' + identity).encode()).hexdigest()[:16], 16)
    probability = bootstrap_probabilities(values, seed=label_seed,
        temperature=temperature, replicates=replicates).mean(axis=0)
    keys = list(request['questions']['action']['criteria']); target = dict.fromkeys(keys, 0.)
    for action, value in zip(eligible, probability): target[keys[action]] = float(value)
    return target


def source_records(spec, config, inputs):
    """Bind the complete native audit before reading any private future labels."""
    if spec['expected_phase'] not in PHASES:
        raise ValueError('Only closed/open attacking additions are supported')
    directory = Path(spec['path']).resolve()
    protocol = json.loads((directory / 'protocol.json').read_text()); source = protocol['config']
    expected_mode = 'attacking_' + spec['expected_phase']
    if (source['decision_filter'] != expected_mode or source['train_scopes'] != ['ordinary']
            or source['reward'] != config['reward'] or source['worlds'] != config['worlds']):
        raise ValueError('Registered attacking source, reward and futures required')
    # This validates all source rows and all eligible alternatives, not a sample.
    normalized = list(audited_records(spec, dict(reward=config['reward'],
        target_temperature=config['temperature'],
        soft_target_mode='outcome_only_win_risk_compatible_masked_v3'), inputs))
    targets = {}
    for path in sorted((directory / 'walls-private').glob('*.pt')):
        if inputs.get(str(path)) != digest(path): raise ValueError('Future shard was not in the complete audit')
        payload = torch.load(path, map_location='cpu', weights_only=False); groups = defaultdict(list)
        for row in payload['rows']:
            groups[(row['episode_key'][0], row['episode_key'][1], row['request_sha256'])].append(row)
        selected, _ = canonical_groups(groups.values(), spec.get('physical_position_canonicalization'))
        for rows in selected:
            first = rows[0]; group = 'native:' + str(first['episode_key'][0])
            key = position(first['public_request'], group)
            if key in targets: raise ValueError('Physical position repeats in the future source')
            targets[key] = bootstrap_target(rows, source, config['label_seed'],
                config['temperature'], config['bootstrap_replicates'])
    used = set()
    for split, record in normalized:
        key = position(record, record['_meta']['group_id'])
        if key not in targets or key in used: raise ValueError('Audited and bootstrapped positions differ')
        question = record['questions']['action']; target = targets[key]
        question.update(target=target, label=max(target, key=target.get))
        used.add(key); yield split, record
    if used != set(targets): raise ValueError('Bootstrap omitted audited source positions')


def append_prepared(parent, additions, out, quotas, caps, minimum_walls, blocked=()):
    """Keep original bytes as prefixes and reject duplicates or split migration."""
    if any(set(spec) != PHASES for table in (quotas, caps, minimum_walls) for spec in table.values()):
        raise ValueError('Explicit two-phase split budgets required')
    if any(set(table) != {'train', 'validation'} for table in (quotas, caps, minimum_walls)):
        raise ValueError('Explicit train/validation budgets required')
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
            for table in (quotas, caps, minimum_walls) for spec in table.values() for v in spec.values()):
        raise ValueError('Positive integer quotas, caps and wall coverage required')
    groups = {s: set() for s in quotas}; positions = set(); identifiers = set(); original = {}
    for split in quotas:
        path = Path(parent['files'][split]['path']); original[split] = path.read_bytes()
        records = 0
        for line in original[split].splitlines():
            if not line.strip(): continue
            records += 1
            record = json.loads(line); meta = record['_meta']; pos = position(record, meta['group_id'])
            if pos in positions or meta['id'] in identifiers: raise ValueError('Parent positions or IDs repeat')
            positions.add(pos); identifiers.add(meta['id']); groups[split].add(meta['group_id'])
        if records != parent['files'][split]['records']:
            raise ValueError('Parent record count differs from admitted receipt')
    selected = {s: [] for s in quotas}; counts = Counter(); walls = defaultdict(Counter)
    for split, record in additions:
        if split not in quotas: raise ValueError('Declared addition split required')
        meta = record['_meta']; route = phase(record); question = record['questions']['action']
        if (route not in PHASES or meta['phase'] != route or meta['label_type'] != 'paired_future_masked'
                or question.get('src') != 'paired_future_masked'
                or not is_attack_request(record, opened=route == 'open_discard')):
            raise ValueError('Only complete non-threat attacking MC additions are allowed')
        target = question['target']; keys = list(question['criteria'])
        if (set(target) != set(keys) or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                or not math.isfinite(v) or v < 0 for v in target.values())
                or abs(sum(target.values()) - 1.) > 1e-10
                or {i for i, k in enumerate(keys) if target[k] > 0} != set(request_eligible(record))
                or question['label'] != max(keys, key=target.__getitem__)):
            raise ValueError('Targets must cover exactly the served candidates')
        group = meta['group_id']; pos = position(record, group)
        if pos in positions or meta['id'] in identifiers: raise ValueError('Addition repeats a position or ID')
        if split == 'train' and group in blocked: raise ValueError('Historical held group cannot enter training')
        positions.add(pos); identifiers.add(meta['id']); groups[split].add(group)
        selected[split].append(record); counts[split, route] += 1; walls[split, route][group] += 1
    if groups['train'] & groups['validation']: raise ValueError('Whole groups cross train and validation')
    for split in quotas:
        for route in PHASES:
            observed = walls[split, route]
            if (counts[split, route] != quotas[split][route] or len(observed) < minimum_walls[split][route]
                    or max(observed.values(), default=0) > caps[split][route]):
                raise ValueError('New data fails its registered count, wall cap or coverage')
    out = Path(out); out.mkdir(parents=True, exist_ok=False); files = {}
    for split in quotas:
        path = out / (split + '.jsonl')
        with path.open('xb') as stream:
            stream.write(original[split])
            if original[split] and not original[split].endswith(b'\n'): stream.write(b'\n')
            for record in sorted(selected[split], key=lambda r: r['_meta']['id']):
                stream.write((json.dumps(record, ensure_ascii=False) + '\n').encode())
        n = parent['files'][split]['records'] + len(selected[split])
        files[split] = dict(path=str(path.resolve()), sha256=digest(path), records=n,
            groups=len(groups[split]), unique_positions=n)
    summary = {s: {p: dict(records=counts[s, p], walls=len(walls[s, p]),
        maximum_records_per_wall=max(walls[s, p].values())) for p in sorted(PHASES)} for s in quotas}
    return files, summary


def prepare(config):
    if config['policy'] != POLICY or config['bootstrap_replicates'] != 256:
        raise ValueError('Explicit audited paired-bootstrap extension required')
    inputs = {}; parent_path = bind(inputs, config['parent_data_receipt'], config['parent_data_receipt_sha256'])
    parent = json.loads(parent_path.read_text())
    if parent['status'] != 'balanced_raw_data_ready' or not parent['requires_masked_soft_target_loss']:
        raise ValueError('Admitted masked parent data required')
    for spec in parent['files'].values(): bind(inputs, spec['path'], spec['sha256'])
    catalog = bind(inputs, config['historical_catalog_index'], config['historical_catalog_sha256'])
    db = sqlite3.connect('file:' + str(catalog) + '?mode=ro', uri=True)
    blocked = {r[0] for r in db.execute("SELECT DISTINCT group_key FROM groups WHERE split!='train'")}; db.close()
    bind(inputs, __file__)
    sources = config['sources']
    if len(sources) != 2 or {s['expected_phase'] for s in sources} != PHASES:
        raise ValueError('Exactly two separately audited attacking phases required')
    additions = [item for spec in sources for item in source_records(spec, config, inputs)]
    for path, checksum in inputs.items():
        if digest(path) != checksum: raise ValueError('Extension input changed')
    files, summary = append_prepared(parent, additions, config['out'], config['quotas'],
        config['whole_wall_caps'], config['minimum_walls'], blocked)
    for path, checksum in inputs.items():
        if digest(path) != checksum: raise ValueError('Extension input changed during materialization')
    out = Path(config['out']); result = copy.deepcopy(parent)
    for key in ('fixed_full_validation', 'fixed_validation_reference', 'strata'): result.pop(key, None)
    result.update(config=config, files=files, attacking_extension=summary,
        training_target_policy=POLICY, parent_training_and_validation_bytes_preserved_as_prefix=True,
        fixed_validation_reference=dict(**parent['files']['validation'], original_bytes_preserved_as_prefix=True),
        expert_labels_changed=False, new_held_targets_added=True, new_simulations=0,
        optimizer_updates=0, raw_records_public=False)
    write(out / 'inputs-private.json', inputs)
    result['target_derivation_inputs_sha256'] = digest(out / 'inputs-private.json')
    write(out / 'result.json', result); write(out / 'status.json', dict(status=result['status'], extension=summary))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True)
    result = prepare(json.loads(parser.parse_args().config.read_text()))
    print(json.dumps(dict(status=result['status'], files=result['files'], extension=result['attacking_extension'])))


if __name__ == '__main__': main()
