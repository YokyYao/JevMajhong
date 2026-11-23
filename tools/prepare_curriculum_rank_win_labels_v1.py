"""Freeze all public-proxy win views and matched leading controls for labels."""
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_rank_win_teacher_v1 import METHOD

PUBLIC_FILES = (
    'tools/curriculum_rank_win_teacher_v1.py', 'tools/prepare_curriculum_rank_win_labels_v1.py',
    'tools/collect_curriculum_rank_win_labels_v1.py', 'tools/merge_curriculum_rank_win_labels_v1.py',
    'tests/core/test_curriculum_rank_win_teacher_v1.py',
    'configs/training/curriculum-rank-win-labels-v1.json', 'cluster/curriculum-rank-win-labels-v1.slurm',
    'docs/experiments/plans/curriculum-rank-win-labels-20261123.md',
)
HELPERS = ('curriculum_terminal_rank_priority_v1.py', 'curriculum_public_win_planner_v1.py',
    'curriculum_observer_rollout_v1.py', 'curriculum_terminal_scout_stats.py',
    'audit_curriculum_reused_terminal_v1.py', 'curriculum_win_context.py', 'curriculum_endgame.py',
    'curriculum_native_decisions.py', 'curriculum_native_worlds.py', 'curriculum_decision_history.py',
    'curriculum_hierarchical_prior.py', 'curriculum_public_inventory.py', 'curriculum_public_worlds.py',
    'curriculum_decision_rollout.py', 'curriculum_endgame_ledger.py', 'audit_curriculum_history_worlds.py')


def prepare(config):
    gpu_only(); c = read(config)
    if (c['method'] != METHOD or c['expected_selected'] != {'train': 218, 'validation': 45}
            or c['worlds_per_block'] != 16 or len(c['outer_seeds']) != 2 or c['bootstrap_samples'] != 512
            or c['future_worlds'] != 4 or c['maximum_future_plans_per_outer_arm'] != 4
            or c['views_per_shard'] != 64 or c['shards'] != 5 or c['student_fit']
            or c['uncertain_rows_filtered'] or c['fourth_escape_veto'] or c['positive_confidence_gate']):
        raise ValueError('Frozen rank32 label recipe required')
    root = ROOT/c['data_root']; audit_path = root/'input-audit-v1/summary-private.json'; audit = read(audit_path)
    original = read(root/'input-audit-v1/completion-binding-private.json')
    files = dict(original['files'])
    if audit['status'] != 'reused_terminal_native_encoder_split_and_public_score_audit_passed':
        raise ValueError('Actual completed native/encoder/score/split input audit required')
    reg_path = ROOT/c['source_registration']; reg = read(reg_path)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.base_lora import package_digest
    model = bundle(Path(reg['curriculum']['reference_checkpoint']))
    if (model['identity_sha256'] != reg['curriculum']['reference_identity']
            or package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']):
        raise ValueError('Frozen v8 continuation and pinned source required')
    paths = [config, reg_path, audit_path, root/'input-audit-v1/completion-binding-private.json',
        root/'input-audit-v1/eligible-small-win-proxies-private.json']
    paths += [ROOT/name for name in PUBLIC_FILES]
    paths += [ROOT/'tools'/name for name in HELPERS]
    paths += [Path(model['directory'])/name for name in model['files']]
    for name in PUBLIC_FILES:
        if name.endswith('.py'): compile((ROOT/name).read_text(), str(ROOT/name), 'exec')
    ready_path = root/'collection-v1/summary-private.json'; ready = read(ready_path); paths.append(ready_path)
    rows = {}; split_ids = {}
    for split in ('train', 'validation'):
        path = ROOT/ready['files'][split]['path']; paths.append(path)
        if sha(path) != ready['files'][split]['sha256']: raise ValueError('Source collection changed')
        pool = [json.loads(line) for line in path.open()]
        for row in pool:
            if row['_meta']['id'] in rows or row['_meta']['split'] != split: raise ValueError('Duplicate/wrong-split source row')
            rows[row['_meta']['id']] = row
        split_ids[split] = {r['_meta']['id'] for r in pool}
    selected = {}; controls = {(r['_meta']['physical_id'], r['_meta']['score_context']): r for r in rows.values()}
    eligible = read(root/'input-audit-v1/eligible-small-win-proxies-private.json')
    physical = {}
    for item in eligible:
        row = rows[item['id']]; m = row['_meta']; s = row['state']
        if (m['category'] != 'win' or m['score_context'] not in c['score_contexts']
                or item['split'] != m['split'] or m['placement']['rank'] not in (3, 4)
                or not item['score']['supported'] or item['score']['own_income'] > 3000
                or item['score']['immediate_rank'] <= 2):
            raise ValueError('Public insufficient-small-win proxy changed')
        selected[m['id']] = row
        physical[m['physical_id']] = m['split']
    for pid, split in physical.items():
        for context in c['leading_controls']:
            row = controls[pid, context]; m = row['_meta']
            if m['split'] != split or m['placement']['rank'] > 2: raise ValueError('Matched public leading control differs')
            selected[m['id']] = row
    specs = [dict(id=k, split=r['_meta']['split'], group=r['_meta']['group_id'],
        physical_id=r['_meta']['physical_id'], context=r['_meta']['score_context'],
        public_rank=r['_meta']['placement']['rank']) for k, r in selected.items()]
    specs.sort(key=lambda v: (0 if v['split'] == 'train' else 1, v['id']))
    counts = Counter(s['split'] for s in specs)
    groups = {split: {s['group'] for s in specs if s['split'] == split} for split in counts}
    if (dict(counts) != c['expected_selected'] or {s: len(v) for s, v in groups.items()} != c['expected_families']
            or groups['train'] & groups['validation']):
        raise ValueError('All263 fixed views and57 original split families required')
    # Check the current actual 3A/3B replay partitions, beyond the older input audit.
    current = read(ROOT/'runs/curriculum-yaku-scenarios-v1/data-v2/data-ready-private.json')
    for split in ('train', 'validation'):
        path = ROOT/current['files'][split]['path']; paths.append(path)
        if sha(path) != current['files'][split]['sha256']: raise ValueError('Current replay data changed')
        opposite = groups['validation' if split == 'train' else 'train']
        if any(json.loads(line)['_meta']['group_id'] in opposite for line in path.open()):
            raise ValueError('Terminal source family overlaps opposite current replay split')
    active_path = ROOT/'runs/curriculum-progression-v1/launch-binding-private.json'
    active = read(active_path)['files']; paths.append(active_path)
    for path in paths: files[str(path.resolve())] = sha(path)
    if any(sha(p) != h for p, h in {**files, **active}.items()): raise ValueError('Immutable source or active3A changed')
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    result = dict(status='all263_public_rank_win_views_frozen_before_values', config=c,
        selected=specs, source_registration=str(reg_path.resolve()), source_files=ready['files'],
        files=files, active_fit_files=active, continuation_model=model,
        expected_selected=dict(counts), expected_families={s: len(v) for s, v in groups.items()},
        no_future_outcomes_used_for_selection=True, labels_generated=0, student_updates=0,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
    write(out/'registered-private.json', result)
    write(out/'prepare-summary-public.json', {k: result[k] for k in (
        'status', 'expected_selected', 'expected_families', 'no_future_outcomes_used_for_selection',
        'labels_generated', 'student_updates', 'job_id', 'node')})
    print(json.dumps(read(out/'prepare-summary-public.json')), flush=True)


if __name__ == '__main__': prepare(ROOT/'configs/training/curriculum-rank-win-labels-v1.json')
