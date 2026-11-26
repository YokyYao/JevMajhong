"""Prepare12-source replay plus actual advanced choices; no parent or fit admission."""
import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.collect_curriculum_endgame_cases import sha
from tools.curriculum_basic_repair import MASS as BASIC_MASS
from tools.build_curriculum_stage1 import public_input

MASS={**{'basic:'+key:.65*value for key,value in BASIC_MASS.items()},
      **{'advanced:'+source+':'+phase:.35/6 for source in ('legacy','teacher') for phase in ('post_meld','call','kan')}}


def layout(counts):
    if set(counts)!=set(MASS) or any(type(n) is not int or n<=0 for n in counts.values()):raise ValueError('All12 actual source strata required')
    n=sum(counts.values());weights={key:MASS[key]*n/counts[key] for key in MASS}
    if max(weights.values())>16:raise ValueError('Proposed per-record loss cap16 exceeded')
    for size in range(192,513):
        updates=math.ceil(n/size);tail=n-(updates-1)*size
        if tail>=.75*size and min(counts.values())>=updates:return dict(accumulation=size,updates_per_epoch=updates,example_weights=weights)
    raise ValueError('No count-only accumulation retains all12 source strata without copies')


def build(basic_root,advanced_root,out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    read=lambda p:json.loads(p.read_text());basic=read(basic_root/'data-ready.json');advanced=read(advanced_root/'summary-private.json')
    if basic['status']!='curriculum_basic_repair_six_source_data_ready' or advanced['status']!='advanced_teacher_native_normalized_encoder_basic_and_ancestral_audit_passed':
        raise ValueError('Complete admitted basic/advanced inputs required')
    binding={}
    for record in (read(basic_root/'source-binding-private.json'),read(advanced_root/'audit-binding-private.json')):
        for path,digest in record['files'].items():
            if sha(path)!=digest:raise ValueError('Source/audit binding changed')
            binding[path]=digest
    old_path=ROOT/'runs/curriculum-reproduction-v1/clean-data-v1/data-ready.json';old=read(old_path)
    counts={};files={};groups={};ids={};out.mkdir(parents=True,exist_ok=False)
    for split in ('train','validation'):
        def load(spec):
            path=ROOT/spec['path']
            if sha(path)!=spec['sha256']:raise ValueError('Audited input file changed')
            binding[str(path.resolve())]=spec['sha256']
            return [json.loads(s) for s in path.open()]
        replay=load(basic['stages']['stage1'][split]);legacy=load(old['stages']['stage2'][split]);teacher=load(advanced['files'][split])
        replay_ids={r['_meta']['id'] for r in replay};rows=[];counter=Counter()
        for role,pool in (('basic',replay),('legacy',legacy),('teacher',teacher)):
            for row in pool:
                if role=='legacy' and row['_meta']['id'] in replay_ids:continue
                if role=='basic':key='basic:'+row['questions']['action']['src'].removeprefix('curriculum1:')
                elif role=='legacy':
                    category={'open_discard':'post_meld','call_window':'call','kan_window':'kan'}.get(row['_meta']['phase'])
                    if category is None:raise ValueError('Unexpected nonreplay legacy phase')
                    key='advanced:legacy:'+category
                else:key='advanced:teacher:'+row['_meta']['curriculum_stratum']
                if key not in MASS:raise ValueError('Undeclared calls-repair stratum')
                item=copy.deepcopy(row);item['questions']['action']['src']='curriculum2:'+key
                if public_input(item)!=public_input(row) or any(item['questions']['action'].get(k)!=row['questions']['action'].get(k) for k in ('label','target')):
                    raise ValueError('Replay/source tagging changed input or supervision')
                rows.append(item);counter[key]+=1
        byid={r['_meta']['id']:r for r in rows}
        if len(byid)!=len(rows):raise ValueError('Duplicate child decision IDs')
        from tools.curriculum_parent import require_replay
        require_replay(replay,rows)
        groups[split]={r['_meta']['group_id'] for r in rows};ids[split]=set(byid);counts[split]=dict(counter)
        if set(counter)!=set(MASS):raise ValueError('A declared source/phase is missing')
        path=out/f'{split}.jsonl'
        with path.open('x') as stream:
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        files[split]=dict(path=str(path.relative_to(ROOT)),records=len(rows),groups=len(groups[split]),sha256=sha(path))
    reserved=set(read(basic_root/'reserved-public-groups.json'))
    if ids['train']&ids['validation'] or groups['train']&(groups['validation']|reserved):raise ValueError('Held-out/reserved child leakage')
    proposed=layout(counts['train']);shutil.copyfile(basic_root/'reserved-public-groups.json',out/'reserved-public-groups.json')
    from tools.curriculum_course_order import course_order
    import random
    categories=[key for key,n in counts['train'].items() for _ in range(n)]
    for epoch in range(2):
        order=course_order(categories,random.Random(2026112209+epoch),proposed['accumulation'])
        if sorted(order)!=list(range(len(categories))):raise ValueError('Incomplete child-source permutation')
    for path in (Path(__file__),ROOT/'docs/experiments/plans/curriculum-calls-repair-data-20261122.md',old_path,
        basic_root/'data-ready.json',advanced_root/'summary-private.json',ROOT/'tools/curriculum_course_order.py'):
        binding[str(path.resolve())]=sha(path)
    if not all(sha(p)==h for p,h in binding.items()):raise ValueError('Source inputs changed during preparation')
    result=dict(status='calls_repair12_source_data_prepared_not_parent_or_fit_admitted',files=files,categories=counts,
        loss_mass=MASS,example_weights=proposed['example_weights'],count_only_accumulation=proposed['accumulation'],
        proposed_updates_per_epoch=proposed['updates_per_epoch'],proposed_epochs=2,all_basic_replay_rows_and_labels_retained=True,
        required_input_schema=None,source_specific_basic_parent_seal_required=True,parent_model_admitted=False,
        additional_independent_teacher_families=0,train_validation_group_overlap=0,train_validation_id_overlap=0,
        no_record_copies=True,no_observed_action_relabelling=True,student_optimizer_updates=0,raw_private=True,
        fit_registration_still_requires_actual_parent_quality_and_frozen_optimizer_cadence=True)
    for name,value in (('data-ready-private.json',result),('preparation-binding-private.json',dict(files=binding))):
        with (out/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--basic',type=Path,required=True);p.add_argument('--advanced',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.basic.resolve(),a.advanced.resolve(),a.out.resolve())
