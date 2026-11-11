"""Prepare unused requests from completed training; never start or select a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from call_policy import digest, request_hash
from checkpoint_identity import completed_checkpoint_files
from feature_continuation import remaining_indices, verify_preparation
from jevany.data import load_records
from jevany.resume import dataset_fingerprint


def prepare(config_path, destination):
    config_path = config_path.resolve()
    destination = destination.resolve()
    destination.relative_to(ROOT / 'data')
    if destination.exists():
        raise ValueError('Use a new immutable data destination')
    cfg = json.loads(config_path.read_text())
    out, data = ROOT / cfg['out'], ROOT / cfg['data']
    preparation = json.loads((out / 'preparation.json').read_text())
    verify_preparation(ROOT, cfg, preparation)
    result = json.loads((out / 'result.json').read_text())
    metrics = json.loads((out / 'training-run/training_metrics.json').read_text())
    audit = json.loads((out / 'training-completion-audit.json').read_text())
    step = result['optimizer_steps']
    if (audit['status'] != 'passed_complete_training_audit'
            or audit['optimizer_steps'] != step or metrics['optimizer_steps'] != step
            or audit['human_test_opened'] or result['human_test_opened']):
        raise ValueError('Completed, audited training required')
    checkpoint = out / 'training-run/checkpoints' / f'step-{step:06d}'
    checkpoint_files = completed_checkpoint_files(checkpoint, step)
    state = torch.load(checkpoint / 'trainer_state.pt', map_location='cpu', weights_only=True)
    if (state['records_seen'] != metrics['records_seen']
            or state['records_seen'] != audit['records_seen']
            or state['records_seen'] != step * metrics['global_effective_batch']
            or metrics['rejected_records'] or metrics['truncated_records']
            or metrics['distributed_padding_records']):
        raise ValueError('Final cursor, admission, dataset or metrics differ')
    lines = (data / 'train.jsonl').read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    if any(not line.endswith(b'\n') for line in lines):
        raise ValueError('Source contains an incomplete record')
    # The trainer hashes loaded records, including loader defaults, not JSONL bytes.
    loaded = load_records(data / 'train.jsonl')
    if (dataset_fingerprint(loaded) != state['dataset_sha256']
            or [r['_meta']['id'] for r in loaded] != [r['_meta']['id'] for r in rows]):
        raise ValueError('Trainer dataset fingerprint or source order differs')
    order = state['order'].tolist()
    remaining = remaining_indices(order, len(rows), epoch=state['epoch'],
        next_microbatch=state['next_microbatch'], records_seen=state['records_seen'],
        batch=state['args']['batch'], world_size=state['world_size'],
        p_none_pair=state['args']['p_none_pair'])
    used = order[:state['next_microbatch']]
    identities, groups = set(), set()
    for row in rows:
        meta = row['_meta']
        if (meta['split'] != 'train' or meta['view'] != 'native'
                or meta['id'] in identities or meta['text_sha256'] != request_hash(row)):
            raise ValueError('Foreign, repeated or changed source request')
        identities.add(meta['id']); groups.add(meta['group_id'])
    if (len(rows) != preparation['records']['train']
            or groups != set(preparation['matches']['train'])):
        raise ValueError('Source coverage differs from registration')
    # Preserve source order and exact bytes. Future training chooses its own order.
    remaining = sorted(remaining)
    unused_ids = {rows[i]['_meta']['id'] for i in remaining}
    used_ids = {rows[i]['_meta']['id'] for i in used}
    if used_ids & unused_ids or len(unused_ids) + len(used_ids) != len(rows):
        raise ValueError('Seen and unused identities do not partition the source')
    counts = lambda indices, key: dict(Counter(key(rows[i]) for i in indices))
    phase = lambda row: row['_meta']['phase']
    action = lambda row: row['questions']['action']['criteria'][row['questions']['action']['label']]['type']
    observed = audit['exposure'][str(step)]
    if (counts(used, phase) != observed['phases']
            or counts(used, action) != observed['actions']):
        raise ValueError('Exposure composition differs from completed audit')
    pinned = [config_path, out / 'preparation.json', out / 'result.json',
        out / 'training-completion-audit.json', out / 'training-run/training_metrics.json',
        checkpoint / 'trainer_state.pt', data / 'train.jsonl', data / 'validation.jsonl',
        Path(__file__).resolve()]
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in pinned}
    report = {'status': 'prepared_only_not_trained',
        'source_config': str(config_path.relative_to(ROOT)),
        'source_final_optimizer_step': step,
        'exclusion_basis': 'All actual exposures through the last completed checkpoint, regardless of selected model',
        'source_pool_requests': len(rows), 'excluded_actual_exposures': len(used),
        'remaining_requests': len(remaining), 'request_overlap_with_exposures': 0,
        'source_train_matches': len(groups),
        'remaining_train_matches': len({rows[i]['_meta']['group_id'] for i in remaining}),
        'new_independent_matches_added': 0,
        'phases': counts(remaining, phase), 'actions': counts(remaining, action),
        'source_indices': remaining, 'source_files': source_hashes,
        'source_checkpoint_files': checkpoint_files,
        'human_source_horizon': 'unknown', 'human_test_opened': False,
        'validation': 'Already-opened source validation copied byte-for-byte; not independent',
        'model_initialization': None, 'optimizer_updates': 0,
        'training_authorized_by_this_manifest': False,
        'next': 'Choose a frozen warm start and training protocol only after complete score evidence'}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + '.preparing-', dir=destination.parent))
    try:
        (temporary / 'train.jsonl').write_bytes(b''.join(lines[i] for i in remaining))
        shutil.copyfile(data / 'validation.jsonl', temporary / 'validation.jsonl')
        report['files'] = {s: digest(temporary / (s + '.jsonl')) for s in ('train', 'validation')}
        if report['files']['validation'] != preparation['files']['validation']:
            raise ValueError('Copied validation differs')
        (temporary / 'pool-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        if any(digest(ROOT / name) != value for name, value in source_hashes.items()):
            raise ValueError('Source changed during preparation')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {k: report[k] for k in ('status', 'source_final_optimizer_step',
        'source_pool_requests', 'excluded_actual_exposures', 'remaining_requests',
        'request_overlap_with_exposures', 'source_train_matches', 'remaining_train_matches',
        'new_independent_matches_added', 'phases', 'actions', 'optimizer_updates', 'human_test_opened')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-config', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_config, args.destination)), flush=True)


if __name__ == '__main__':
    main()
