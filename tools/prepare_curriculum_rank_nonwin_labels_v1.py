"""Freeze late non-win source families and paired placement contexts before values."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.curriculum_rank_nonwin_teacher_v1 import METHOD

PUBLIC_FILES = ('tools/curriculum_rank_nonwin_teacher_v1.py', 'tools/prepare_curriculum_rank_nonwin_labels_v1.py',
    'tools/collect_curriculum_rank_nonwin_labels_v1.py', 'tools/merge_curriculum_rank_nonwin_labels_v1.py',
    'tests/core/test_curriculum_rank_nonwin_teacher_v1.py',
    'configs/training/curriculum-rank-nonwin-labels-v1.json',
    'cluster/curriculum-rank-nonwin-labels-v1.slurm',
    'docs/experiments/plans/curriculum-rank-nonwin-labels-20261123.md')


def prepare():
    gpu_only(); config = ROOT/'configs/training/curriculum-rank-nonwin-labels-v1.json'; c = read(config)
    if (c['method'] != METHOD or c['candidate_cap'] != 4 or c['worlds_per_block'] != 16
            or c['expected_views'] != {'train':96,'validation':24}
            or c['physical_families_per_category'] != {'train':8,'validation':2}
            or c['contexts'] != ['hold_second','third_gap3000','fourth_gap5000','fourth_gap13000']
            or not c['unique_families_across_categories'] or c['shards'] != 3):
        raise ValueError('Exact public non-win cohort and fixed32-world recipe required')
    audited_path = ROOT/c['data_root']/'input-audit-v1/summary-private.json'; audited = read(audited_path)
    if audited['status'] != 'reused_terminal_native_encoder_split_and_public_score_audit_passed':
        raise ValueError('Completed native/source-group/tokenizer input audit required')
    original_binding_path = audited_path.parent/'completion-binding-private.json'
    files = dict(read(original_binding_path)['files'])
    source_path = ROOT/c['source_registration']; reg = read(source_path)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.base_lora import package_digest
    if (package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']
            or package_digest(Path(reg['training_config']['upstream_source'])/'jevany') != reg['training_config']['upstream_source_sha256']):
        raise ValueError('Pinned model and upstream runtime changed')
    from tools.prepare_curriculum_rank_win_labels_v1 import HELPERS
    paths = [config, source_path, audited_path, original_binding_path]
    paths += [ROOT/name for name in PUBLIC_FILES]+[ROOT/'tools'/name for name in HELPERS]
    paths += [ROOT/'tools/curriculum_rank_win_teacher_v1.py', ROOT/'tools/prepare_curriculum_rank_win_labels_v1.py']
    for name in PUBLIC_FILES:
        if name.endswith('.py'): compile((ROOT/name).read_text(), str(ROOT/name), 'exec')
    ready_path = ROOT/c['data_root']/'collection-v1/summary-private.json'; ready = read(ready_path); paths.append(ready_path)
    pools = {}; selected = []; coverage = {}
    for split in ('train','validation'):
        path = ROOT/ready['files'][split]['path']; paths.append(path)
        if sha(path) != ready['files'][split]['sha256']: raise ValueError('Original unlabelled source changed')
        rows = [json.loads(line) for line in path.open()]; pools[split] = {r['_meta']['id']:r for r in rows}
        physical = {}
        for row in rows:
            m = row['_meta']; q = row['questions']['action']
            if (m['category'] in c['categories'] and len(q['criteria']) >= 2
                    and not any(v['type']=='hora' for v in q['criteria'].values())):
                physical.setdefault(m['physical_id'], {})[m['score_context']] = row
        used = set()
        for category in c['categories']:
            candidates = [(pid, views) for pid, views in physical.items()
                if set(c['contexts']) <= set(views)
                and views[c['contexts'][0]]['_meta']['category'] == category]
            candidates.sort(key=lambda p: hashlib.sha256(('rank-nonwin-v1:'+p[0]).encode()).hexdigest())
            available = Counter()
            for pid, views in candidates:
                for op in {v['type'] for v in views[c['contexts'][0]]['questions']['action']['criteria'].values()}:
                    available[op] += 1
            coverage[split+':'+category] = dict(available)
            # Public-operation priority reserves scarce kan/riichi opportunities
            # when available, without choosing actions or looking at outcomes.
            priority = []; reserved = set(used)
            for op in c['preferred_public_operations'][category]:
                matches = [(pid, views) for pid, views in candidates
                    if views[c['contexts'][0]]['_meta']['group_id'] not in reserved
                    and any(v['type'] == op for v in views[c['contexts'][0]]['questions']['action']['criteria'].values())]
                if matches:
                    priority.append(matches[0]); reserved.add(matches[0][1][c['contexts'][0]]['_meta']['group_id'])
            candidates = priority+candidates
            taken = 0
            for pid, views in candidates:
                first = views[c['contexts'][0]]; group = first['_meta']['group_id']
                if group in used: continue
                used.add(group); taken += 1
                for context in c['contexts']:
                    row = views[context]; m = row['_meta']
                    if m['split'] != split or m['group_id'] != group: raise ValueError('Context lineage changed')
                    selected.append(dict(id=m['id'], split=split, group=group, physical_id=pid,
                        category=category, context=context, public_rank=m['placement']['rank']))
                if taken == c['physical_families_per_category'][split]: break
            if taken != c['physical_families_per_category'][split]:
                raise ValueError('Insufficient distinct predeclared phase families')
    selected.sort(key=lambda s:(s['split']!='train', s['category'], s['id']))
    counts = Counter(s['split'] for s in selected)
    groups = {split:{s['group'] for s in selected if s['split']==split} for split in ('train','validation')}
    if (dict(counts) != c['expected_views'] or {s:len(v) for s,v in groups.items()} != c['expected_families']
            or groups['train'] & groups['validation']):
        raise ValueError('Exact120 views and original24/6 source families required')
    current_path = ROOT/'runs/curriculum-yaku-scenarios-v1/data-v2/data-ready-private.json'; current=read(current_path); paths.append(current_path)
    for split in ('train','validation'):
        path = ROOT/current['files'][split]['path']; paths.append(path)
        if sha(path) != current['files'][split]['sha256']: raise ValueError('Current replay changed')
        other = groups['validation' if split=='train' else 'train']
        if any(json.loads(line)['_meta']['group_id'] in other for line in path.open()):
            raise ValueError('Opposite current replay family overlap')
    active_path=ROOT/'runs/curriculum-progression-v1/launch-binding-private.json'
    active=read(active_path)['files']; paths.append(active_path)
    model=bundle(Path(reg['curriculum']['reference_checkpoint']))
    if model['identity_sha256'] != reg['curriculum']['reference_identity']: raise ValueError('Frozen v8 changed')
    paths += [Path(model['directory'])/name for name in model['files']]
    for path in paths: files[str(path.resolve())]=sha(path)
    if any(sha(p)!=h for p,h in {**files,**active}.items()): raise ValueError('Source/active input changed')
    out=ROOT/c['out']; out.mkdir(parents=True,exist_ok=False)
    result=dict(status='public_nonwin120_views_and24_6_families_frozen_before_values',config=c,
        selected=selected,source_files=ready['files'],files=files,active_fit_files=active,
        source_registration=str(source_path.resolve()),continuation_model=model,
        counts=dict(counts),families={s:len(v) for s,v in groups.items()},
        public_operation_available_physical_windows=coverage,
        selected_by_public_metadata_not_simulated_values=True,labels_generated=0,student_updates=0,
        node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'])
    write(out/'registered-private.json',result)
    write(out/'prepare-summary-public.json',{k:result[k] for k in (
        'status','counts','families','selected_by_public_metadata_not_simulated_values',
        'labels_generated','student_updates','node','job_id')})
    print(json.dumps(read(out/'prepare-summary-public.json')),flush=True)


if __name__=='__main__':prepare()
