"""Condition open-discard MC supervision on the existing serving candidates."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path

from jevmahjong.core.io import digest, write
from jevmahjong.policy.ordinary import is_ordinary_request, request_eligible
from .base_lora_pool import position
from .relabel_targets import bind

POLICY = 'open_mc_served_support_conditioning_v1'


def conditional_target(record):
    """Remove excluded mass without inventing values for unknown actions."""
    meta = record['_meta']
    if meta['label_type'] != 'paired_future_masked' or meta['phase'] != 'open_discard':
        raise ValueError('Only open-discard MC training targets may be conditioned')
    state = record['state']
    if (not is_ordinary_request(record)
            or not any(m['type'] not in (3, 'ankan') for m in state['melds'][state['seat']])):
        raise ValueError('Ordinary opened-hand public input required')
    question = record['questions']['action']; keys = list(question['criteria'])
    old = question['target']
    if (set(old) != set(keys) or any(isinstance(v, bool) or not isinstance(v, (int, float))
            or not math.isfinite(v) or v < 0 for v in old.values())
            or abs(sum(old.values()) - 1.) > 1e-10):
        raise ValueError('Finite normalized target over the original menu required')
    allowed = {keys[i] for i in request_eligible(record)}
    if not allowed or any(old[k] <= 0 for k in allowed):
        raise ValueError('Every served action needs an existing positive label')
    mass = sum(old[k] for k in allowed)
    excluded = sum(old[k] for k in keys if k not in allowed)
    target = ({k: old[k] / mass if k in allowed else 0. for k in keys}
        if excluded > 0 else dict(old))
    if abs(sum(target.values()) - 1.) > 1e-10:
        raise ValueError('Conditional target normalization failed')
    return target, dict(excluded_mass=excluded,
        preferred_excluded=max(keys, key=old.__getitem__) not in allowed)


def prepare(config):
    if config['policy'] != POLICY:
        raise ValueError('Explicit served-support conditioning policy required')
    inputs = {}
    parent_path = bind(inputs, config['parent_data_receipt'], config['parent_data_receipt_sha256'])
    parent = json.loads(parent_path.read_text())
    if (parent['status'] != 'balanced_raw_data_ready' or not parent['requires_masked_soft_target_loss']
            or set(parent['files']) != {'train', 'validation'}):
        raise ValueError('Completed admitted masked training data required')
    for spec in parent['files'].values():
        bind(inputs, spec['path'], spec['sha256'])
    from jevmahjong.policy import ordinary
    bind(inputs, __file__); bind(inputs, ordinary.__file__)
    out = Path(config['out']); out.mkdir(parents=True, exist_ok=False)
    stats = Counter(); seen = set()
    with Path(parent['files']['train']['path']).open('rb') as source, (out / 'train.jsonl').open('xb') as target:
        for line in source:
            if not line.strip():
                target.write(line); continue
            row = json.loads(line); pos = position(row, row['_meta']['group_id'])
            if pos in seen:
                raise ValueError('Conditioning requires unique admitted training positions')
            seen.add(pos); stats['training_records'] += 1
            if row['_meta']['label_type'] != 'paired_future_masked' or row['_meta']['phase'] != 'open_discard':
                target.write(line); continue
            proposed, diagnostic = conditional_target(row)
            stats['conditioned_open_mc_records'] += 1
            stats['preferred_excluded'] += diagnostic['preferred_excluded']
            stats['excluded_mass_sum'] += diagnostic['excluded_mass']
            if proposed == row['questions']['action']['target']:
                target.write(line); continue
            stats['changed_target_records'] += 1
            question = row['questions']['action']
            stats['argmax_changes'] += max(proposed, key=proposed.get) != question['label']
            question.update(target=proposed, label=max(proposed, key=proposed.get))
            target.write((json.dumps(row, ensure_ascii=False) + '\n').encode())
    if stats['training_records'] != parent['files']['train']['records']:
        raise ValueError('Conditioning omitted admitted records')
    (out / 'validation.jsonl').write_bytes(Path(parent['files']['validation']['path']).read_bytes())
    for path, checksum in inputs.items():
        if digest(path) != checksum:
            raise ValueError('Conditioning input changed during materialization')
    files = {split: dict(spec, path=str((out / (split + '.jsonl')).resolve()),
        sha256=digest(out / (split + '.jsonl'))) for split, spec in parent['files'].items()}
    result = copy.deepcopy(parent)
    result.update(config=config, files=files, training_target_policy=POLICY,
        support_conditioning=dict(stats), training_position_order_and_IDs_preserved=True,
        expert_labels_changed=False, validation_targets_changed=False,
        public_model_inputs_changed=False, unknown_actions_are_not_negative_labels=True,
        excluded_actions_are_unlabelled=True, new_simulations=0, optimizer_updates=0,
        raw_records_public=False)
    write(out / 'inputs-private.json', inputs)
    result['target_derivation_inputs_sha256'] = digest(out / 'inputs-private.json')
    write(out / 'result.json', result)
    write(out / 'status.json', dict(status=result['status'], **stats))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    result = prepare(json.loads(parser.parse_args().config.read_text()))
    print(json.dumps(dict(status=result['status'], support_conditioning=result['support_conditioning'])))


if __name__ == '__main__': main()
