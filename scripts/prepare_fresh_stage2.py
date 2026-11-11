"""Same-recipe stage2 continuation on entirely new TRAIN match groups."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_shard import split_for_match
from call_option_features import enrich_request
from call_policy import digest, request_hash
from checkpoint_identity import completed_checkpoint_files
from head_rl import write


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text())
    data, out = ROOT / cfg['data'], ROOT / cfg['out']
    if data.exists() or out.exists(): raise ValueError('New immutable data and run required')
    raw_cfg = json.loads((ROOT / cfg['raw_config']).read_text()); raw = ROOT / raw_cfg['data']
    audit = json.loads((raw / 'provenance.json').read_text())
    if (not audit['complete'] or audit['config'] != raw_cfg or audit['model_test_opened']
            or audit['records'] != {'train': 9600, 'validation': 0, 'test': 0}
            or audit['archive_sha256'] != cfg['source_archive_sha256']):
        raise ValueError('TRAIN-only raw data incomplete or source differs')
    if audit['matches']['validation'] or audit['matches']['test']:
        raise ValueError('Unexpected new validation/test match use')
    excluded = set()
    for name in raw_cfg['exclude_provenance']:
        old = json.loads((ROOT / name).read_text())
        excluded.update(m for ms in old['matches'].values() for m in ms)
    new_groups = set(audit['matches']['train'])
    if new_groups & excluded or any(split_for_match(m) != 'train' for m in new_groups):
        raise ValueError('Fresh TRAIN matches overlap prior exposure or holdout split')
    if digest(raw / 'train.jsonl') != audit['files']['train']:
        raise ValueError('Raw TRAIN changed')
    parent_cfg = json.loads((ROOT / cfg['parent_config']).read_text()); parent = ROOT / parent_cfg['out']
    parent_result = json.loads((parent / 'result.json').read_text())
    parent_protocol = json.loads((parent / 'protocol.json').read_text())
    parent_prepared = json.loads((parent / 'preparation.json').read_text())
    if (parent_result['checkpoint'] != cfg['warm_start'] or parent_result['selected_step'] != 200
            or cfg['parent_step'] != 200 or parent_protocol['config'] != parent_cfg):
        raise ValueError('Must warm start exact preserved stage2')
    warm_files = completed_checkpoint_files(ROOT / cfg['warm_start'], 200)
    old_data = ROOT / parent_cfg['data']
    if digest(old_data / 'validation.jsonl') != parent_prepared['files']['validation']:
        raise ValueError('Preserved validation changed')
    validation = [json.loads(l) for l in (old_data / 'validation.jsonl').open()]
    if new_groups & {r['_meta']['group_id'] for r in validation}:
        raise ValueError('Whole-match validation overlap')
    data.mkdir(parents=True); out.mkdir(parents=True)
    ids, groups, phases, actions = set(), set(), Counter(), Counter()
    with (data / 'train.jsonl').open('w') as stream:
        for line in (raw / 'train.jsonl').open():
            r = json.loads(line); meta = r['_meta']
            if meta['split'] != 'train' or meta['view'] != 'native' or meta['group_id'] not in new_groups:
                raise ValueError('Foreign raw TRAIN row')
            if meta['id'] in ids: raise ValueError('Duplicate raw identity')
            ids.add(meta['id']); groups.add(meta['group_id'])
            label = r['questions']['action']['label']; criteria = r['questions']['action']['criteria']
            transformed = enrich_request(r)
            if (list(transformed['questions']['action']['criteria']) != list(criteria)
                    or transformed['questions']['action']['label'] != label):
                raise ValueError('Feature conversion changed actions or expert labels')
            transformed['_meta']['text_sha256'] = request_hash(transformed)
            transformed['_meta']['source_match_horizon'] = 'unknown'
            stream.write(json.dumps(transformed, ensure_ascii=False) + '\n')
            phases[meta['phase']] += 1; actions[criteria[label]['type']] += 1
    if groups != new_groups or dict(phases) != raw_cfg['raw_phase_counts']['train'] or len(ids) != 9600:
        raise ValueError('Fresh match/request/phase coverage incomplete')
    shutil.copyfile(old_data / 'validation.jsonl', data / 'validation.jsonl')
    frozen = set(parent_protocol['code_sha256'])
    frozen.update(['scripts/prepare_fresh_stage2.py', 'scripts/train_feature_continuation.py',
        'feature_continuation.py', 'feature_training_guard.py', 'call_option_features.py',
        'guide_call_features.py', 'scripts/prepare_clean_policy.py', cfg['trainer_config'],
        str(args.config.resolve().relative_to(ROOT)), cfg['raw_config'], cfg['parent_config']])
    frozen.update(str((raw / n).relative_to(ROOT)) for n in ('provenance.json', 'train.jsonl'))
    frozen.update(str((parent / n).relative_to(ROOT)) for n in ('protocol.json', 'result.json', 'preparation.json'))
    frozen.add(str((old_data / 'validation.jsonl').relative_to(ROOT)))
    frozen.update(str((ROOT / cfg['warm_start'] / n).relative_to(ROOT)) for n in (*warm_files, 'trainer_state.pt'))
    frozen.update(str((ROOT / cfg['reference'] / n).relative_to(ROOT)) for n in parent_protocol['reference_files'])
    frozen.update(raw_cfg['exclude_provenance'])
    report = {'complete': True, 'config': cfg, 'model_test_opened': False,
        'records': {'train': len(ids), 'validation': len(validation)},
        'train_actions': dict(actions), 'train_phases': dict(phases), 'train_matches': len(groups),
        'matches': {'train': sorted(groups)}, 'excluded_prior_matches': len(excluded),
        'whole_match_prior_overlap': 0, 'whole_match_train_validation_overlap': 0,
        'new_human_validation_or_test_matches': 0, 'source_match_horizon': 'unknown',
        'interpretation': 'Entirely new TRAIN matches, unchanged task instruction/features/loss/phase mix/optimizer recipe; human source match length not verified',
        'warm_start_files': warm_files, 'reference_files': parent_protocol['reference_files'],
        'files': {s: digest(data / (s + '.jsonl')) for s in ('train', 'validation')},
        'frozen_files': {n: digest(ROOT / n) for n in sorted(frozen)}}
    write(data / 'provenance.json', report); write(out / 'preparation.json', report)
    print(json.dumps({k: report[k] for k in ('complete', 'records', 'train_phases',
        'train_actions', 'train_matches', 'excluded_prior_matches', 'whole_match_prior_overlap')}), flush=True)


if __name__ == '__main__': main()
