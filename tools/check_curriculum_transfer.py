"""Check real replay data/path/sampler compatibility without admitting a fit."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def check(registered,child_data_ready,out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-','login'))
    from tools.curriculum_transfer import read,sha,canonical_protocol
    from tools.curriculum_parent import require_child_input_contract
    from tools.curriculum_course_order import CourseEpochRandom
    from tools.audit_curriculum_accumulation import suggest_accumulation
    reg=read(registered);child=read(child_data_ready);require_child_input_contract(reg,child)
    for path,h in reg['inputs'].items():assert sha(path)==h,path
    for spec in child['files'].values():assert sha(ROOT/spec['path'])==spec['sha256']
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(1)
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.training.initialization import lineage_training_files,check_groups
    from jevany.data import api_request
    assert package_digest(Path(reg['source'])/'jevmahjong')==reg['source_sha256']
    fit=ROOT/reg['curriculum']['out'];original=read(fit/'protocol.json');protocol=canonical_protocol(original)
    lineage=lineage_training_files(protocol)
    groups=check_groups(protocol['trainer_args']['data'],ROOT/child['files']['train']['path'],
        ROOT/child['files']['validation']['path'],lineage[1:])
    load=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines() if s]
    parent=load(protocol['trainer_args']['data']);train=load(ROOT/child['files']['train']['path'])
    validation=load(ROOT/child['files']['validation']['path']);train_byid={r['_meta']['id']:r for r in train}
    assert len(train_byid)==len(train)
    for row in parent:
        assert row['_meta']['id'] in train_byid
        assert api_request(row)==api_request(train_byid[row['_meta']['id']])
        assert row['questions']['action']['label']==train_byid[row['_meta']['id']]['questions']['action']['label']
    protected={r['_meta']['group_id'] for r in load(protocol['trainer_args']['eval_data'])}
    protected.update(r['_meta']['group_id'] for r in validation)
    protected.update(read(ROOT/reg['curriculum']['data_root']/'reserved-public-groups.json'))
    assert not {r['_meta']['group_id'] for r in train}&protected
    categories=[r['questions']['action']['src'].removeprefix('curriculum2:') for r in train]
    counts=Counter(categories);assert dict(counts)==child['categories']['train']
    proposal=suggest_accumulation(len(train),128,min(counts.values()))
    rng=CourseEpochRandom(2026112104,categories,proposal['accumulation'],out.parent/'course-order-check-private')
    order=list(range(len(train)));epoch_proofs=[]
    for epoch in range(2):
        rng.shuffle(order)
        assert Counter(categories[i] for i in order)==counts
        proof=read(rng.receipt_dir/f'epoch-{epoch+1:02d}.json')
        assert all(set(v)==set(counts) for v in proof['source_counts_by_update'])
        epoch_proofs.append(dict(epoch=epoch+1,records=proof['records'],updates=proof['updates'],
            final_records=sum(proof['source_counts_by_update'][-1].values()),all_strata_every_update=True))
    result=dict(status='real_transfer_data_paths_checked_not_parent_admitted',node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
        source_protocol_unchanged=sha(fit/'protocol.json'),ancestral_training_files=lineage,
        relative_input_keys_given_checked_absolute_aliases=sum(not Path(k).is_absolute() for k in original['input_sha256']),
        pinned_legacy_lineage_and_group_checks_passed=True,groups=groups,parent_replay_rows=len(parent),
        parent_replay_labels_and_public_requests_equal=True,protected_group_overlap=0,proposal=proposal,
        two_diagnostic_epoch_receipts=epoch_proofs,student_model_loaded=False,parent_seal_created=False,
        child_fit_admitted=False,optimizer_updates=0,new_training_labels=False,
        complete_qualified_parent_load_and_warm_start_still_pending=True)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--child-data-ready',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();check(a.registered,a.child_data_ready,a.out)
