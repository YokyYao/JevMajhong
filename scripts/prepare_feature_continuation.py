"""Exclude exact prior-stage exposure while preserving human labels and splits."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from checkpoint_identity import completed_checkpoint_files
from feature_continuation import remaining_indices
from head_rl import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    config_path = parser.parse_args().config
    cfg = json.loads(config_path.read_text())
    data, out = ROOT / cfg['data'], ROOT / cfg['out']
    if data.exists() or out.exists():
        raise ValueError('New immutable data and run paths required')
    source_cfg = json.loads((ROOT / cfg['parent_config']).read_text())
    parent = ROOT / source_cfg['out']
    protocol = json.loads((parent / 'protocol.json').read_text())
    result = json.loads((parent / 'result.json').read_text())
    old = json.loads((parent / 'preparation.json').read_text())
    if protocol['config'] != source_cfg or not old['complete'] or old['model_test_opened']:
        raise ValueError('Parent protocol or preparation differs')
    if result['status'] != 'clean_policy_candidate' or result['selected_step'] != cfg['parent_step']:
        raise ValueError('Registered selected parent required')
    if result['checkpoint'] != cfg['warm_start']:
        raise ValueError('Parent checkpoint path changed')
    development = ROOT / cfg['parent_development_out']
    audit = json.loads((development / 'evaluation-audit.json').read_text())
    dev_protocol = json.loads((development / 'protocol.json').read_text())
    gain = audit['overall_paired_gain']['gain']
    if audit['status'] != 'passed_complete' or audit['stage'] != 'development' or not math.isfinite(gain) or gain <= 0:
        raise ValueError('Positive fully audited parent development required')
    if dev_protocol['training_result_sha256'] != digest(parent / 'result.json'):
        raise ValueError('Parent development selected different training result')
    warm = ROOT / cfg['warm_start']
    warm_files = completed_checkpoint_files(warm, cfg['parent_step'])
    if dev_protocol['candidate_files'] != warm_files:
        raise ValueError('Parent development weights differ')
    if protocol['reference_files'] != json.loads((ROOT / cfg['reference_protocol']).read_text())['reference_files']:
        raise ValueError('Reference differs from parent')
    for name, expected in protocol['code_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Parent source changed: ' + name)
    initial = json.loads((parent / 'initialization-audit.json').read_text())
    if not initial['warm_start_exact_adapter_and_head'] or not initial['no_oldB_weights_loaded']:
        raise ValueError('Verified clean-parent initialization required')
    source_data = ROOT / source_cfg['data']
    for split in ('train', 'validation'):
        if digest(source_data / (split + '.jsonl')) != old['files'][split]:
            raise ValueError('Parent data changed')
    admission = json.loads((parent / 'token-admission.json').read_text())
    if admission['rejected_records'] or admission['admitted_records'] != old['records']['train']:
        raise ValueError('Saved order must cover the full unfiltered parent pool')
    import torch
    state = torch.load(warm / 'trainer_state.pt', map_location='cpu', weights_only=True)
    if state['optimizer_step'] != cfg['parent_step']:
        raise ValueError('Parent cursor checkpoint step differs')
    args = state['args']
    remaining = remaining_indices(state['order'].tolist(), old['records']['train'],
        epoch=state['epoch'], next_microbatch=state['next_microbatch'],
        records_seen=state['records_seen'], batch=args['batch'],
        world_size=state['world_size'], p_none_pair=args['p_none_pair'])
    if state['records_seen'] != cfg['exclude_parent_requests']:
        raise ValueError('Parent exposure differs from registered count')
    remaining = set(remaining)
    validation = [json.loads(line) for line in (source_data / 'validation.jsonl').open()]
    validation_ids = {r['_meta']['id'] for r in validation}
    validation_groups = {r['_meta']['group_id'] for r in validation}
    data.mkdir(parents=True)
    out.mkdir(parents=True)
    labels, phases, groups, seen_ids = Counter(), Counter(), set(), set()
    with (data / 'train.jsonl').open('w') as stream:
        source_count = 0
        for index, line in enumerate((source_data / 'train.jsonl').open()):
            source_count += 1
            r = json.loads(line)
            identity, group = r['_meta']['id'], r['_meta']['group_id']
            if identity in seen_ids or identity in validation_ids or group in validation_groups:
                raise ValueError('Duplicate source request or whole-match validation overlap')
            seen_ids.add(identity)
            if index not in remaining:
                continue
            if r['_meta']['view'] != 'native' or r['_meta']['split'] != 'train':
                raise ValueError('Only original native TRAIN requests admitted')
            stream.write(line if line.endswith('\n') else line + '\n')
            q = r['questions']['action']
            labels[q['criteria'][q['label']]['type']] += 1
            phases[r['_meta']['phase']] += 1
            groups.add(group)
    if source_count != old['records']['train'] or sum(labels.values()) != len(remaining):
        raise ValueError('Unseen source coverage incomplete')
    shutil.copyfile(source_data / 'validation.jsonl', data / 'validation.jsonl')
    frozen = set(protocol['code_sha256'])
    frozen.update([cfg['parent_config'], cfg['trainer_config'], str(config_path.resolve().relative_to(ROOT)),
        'scripts/prepare_feature_continuation.py', 'scripts/train_feature_continuation.py',
        'feature_continuation.py', 'feature_training_guard.py'])
    frozen.update(str((parent / name).relative_to(ROOT)) for name in
        ('protocol.json', 'result.json', 'preparation.json', 'initialization-audit.json', 'token-admission.json'))
    frozen.update(str((development / name).relative_to(ROOT)) for name in
        ('protocol.json', 'summary.json', 'evaluation-audit.json'))
    frozen.update(str((source_data / (split + '.jsonl')).relative_to(ROOT)) for split in ('train', 'validation'))
    frozen.update(str((warm / name).relative_to(ROOT)) for name in (*warm_files, 'trainer_state.pt'))
    frozen.update(str((ROOT / cfg['reference'] / name).relative_to(ROOT)) for name in protocol['reference_files'])
    report = {'complete': True, 'config': cfg, 'model_test_opened': False,
        'records': {'train': len(remaining), 'validation': len(validation)},
        'train_actions': dict(labels), 'train_phases': dict(phases),
        'train_matches': len(groups), 'validation_matches': len(validation_groups),
        'excluded_parent_requests': state['records_seen'], 'parent_selected_step': cfg['parent_step'],
        'overlap_with_parent_seen_requests': 0, 'whole_match_train_validation_overlap': 0,
        'scope': 'Unseen request IDs within existing TRAIN matches; not newly independent matches or test data',
        'order': 'Original source order excluding exact saved prior exposure, reshuffled by the new stage seed',
        'warm_start_files': warm_files, 'reference_files': protocol['reference_files'],
        'files': {split: digest(data / (split + '.jsonl')) for split in ('train', 'validation')},
        'frozen_files': {name: digest(ROOT / name) for name in sorted(frozen)}}
    write(data / 'provenance.json', report)
    write(out / 'preparation.json', report)
    print(json.dumps({k: report[k] for k in ('complete', 'records', 'train_actions',
        'train_phases', 'train_matches', 'excluded_parent_requests', 'overlap_with_parent_seen_requests')}))


if __name__ == '__main__':
    main()
