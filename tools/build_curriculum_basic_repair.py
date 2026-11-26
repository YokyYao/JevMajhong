"""Keep complete legacy replay and audited new teacher rows with six loss tags."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.collect_curriculum_endgame_cases import sha
from tools.curriculum_basic_repair import MASS,weights,accumulation
from tools.build_curriculum_stage1 import public_input,stratum


def build(teacher,out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    read=lambda p:json.loads(p.read_text());audited=read(teacher/'summary-private.json')
    if audited['status']!='basic_teacher_native_normalized_encoder_and_available_ancestral_audit_passed':raise ValueError('Complete teacher audit required')
    teacher_binding=read(teacher/'audit-binding-private.json')['files']
    if not all(sha(p)==h for p,h in teacher_binding.items()):raise ValueError('Audited teacher source changed')
    parent=ROOT/'runs/curriculum-reproduction-v1/clean-data-v1';old=read(parent/'data-ready.json')
    out.mkdir(parents=True,exist_ok=False);(out/'stage1').mkdir();files={};counts={};groups={};ids={};binding=dict(teacher_binding)
    for split in ('train','validation'):
        rows=[]
        for source,spec in (('legacy',old['stages']['stage1'][split]),('teacher',audited['files'][split])):
            path=ROOT/spec['path']
            if sha(path)!=spec['sha256']:raise ValueError('Admitted source file changed')
            binding[str(path.resolve())]=spec['sha256']
            for line in path.open():
                row=json.loads(line);item=copy.deepcopy(row);item['questions']['action']['src']='curriculum1:'+source+':'+stratum(row)
                if public_input(item)!=public_input(row) or item['questions']['action']['label']!=row['questions']['action']['label']:
                    raise ValueError('Source tagging changed public input or supervision')
                item['_meta']['basic_repair_source']=source;rows.append(item)
        ids[split]={r['_meta']['id'] for r in rows};groups[split]={r['_meta']['group_id'] for r in rows}
        if len(ids[split])!=len(rows):raise ValueError('Duplicate replay/teacher decision identity')
        counts[split]=dict(Counter(r['questions']['action']['src'].removeprefix('curriculum1:') for r in rows))
        if set(counts[split])!=set(MASS):raise ValueError('A source/class is absent')
        path=out/'stage1'/f'{split}.jsonl'
        with path.open('x') as stream:
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        files[split]=dict(path=str(path.relative_to(ROOT)),records=len(rows),sha256=sha(path),groups=len(groups[split]))
    reserved=set(read(parent/'reserved-public-groups.json'))
    if ids['train']&ids['validation'] or groups['train']&(groups['validation']|reserved):raise ValueError('Held-out source/decision leakage')
    size,updates=accumulation(counts['train']);loss=weights(counts['train'])
    for name in ('reserved-public-groups.json','common-validation.jsonl'):shutil.copyfile(parent/name,out/name)
    paths=[Path(__file__),ROOT/'tools/curriculum_basic_repair.py',ROOT/'docs/experiments/plans/curriculum-basic-repair-fit-20261122.md',
        parent/'data-ready.json',teacher/'summary-private.json',teacher/'audit-binding-private.json']
    binding.update({str(p.resolve()):sha(p) for p in paths})
    if not all(sha(p)==h for p,h in binding.items()):raise ValueError('Mixture source changed')
    ready=dict(status='curriculum_basic_repair_six_source_data_ready',stages={'stage1':files},stage1_strata=counts,
        stage1_loss_mass=MASS,stage1_loss_weights=loss,accumulation=size,updates_per_epoch=updates,epochs=4,
        train_validation_group_overlap=0,train_validation_id_overlap=0,all_legacy_and_teacher_rows_once=True,copies_added=0,
        labels_unchanged=True,model_visible_inputs_unchanged=True,reserved_groups=len(reserved),
        teacher_audit=str(teacher/'summary-private.json'),future_courses_must_retain_all_basic_replay=True,
        win_choice_supervision=False,legal_wins_executed_by_rule=True,student_fit_admitted=False,optimizer_updates=0)
    for name,value in (('data-ready.json',ready),('source-binding-private.json',dict(files=binding))):
        with (out/name).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(ready),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--teacher',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.teacher.resolve(),a.out.resolve())
