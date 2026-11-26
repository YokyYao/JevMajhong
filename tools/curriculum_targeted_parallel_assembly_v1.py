"""Assemble only ended disjoint cases into the original complete27-view shard."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import FunctionType

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools import curriculum_targeted_parallel_worker_v1 as worker
from tools import curriculum_targeted_win_resume_v2 as original
from tools import curriculum_targeted_win_resume_v3 as cached
from tools import curriculum_targeted_win_resumed_pipeline_v3 as pipeline
from tools.capture_curriculum_paired_terminal_v2 import resource,scheduler,terminal_record,fields
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write

FULL=worker.RUN/'full-shard03-v1'
ENTRY=ROOT/'cluster/curriculum-targeted-case-assembly-v1.slurm'
MERGE_ENTRY=ROOT/'cluster/curriculum-targeted-case-merge-v1.slurm'
STATUS='actual_all27_original_targeted_views_assembled_from_ended_disjoint_cases'


def require_terminal(archive=None):
    launch=read(FULL/'registered-private.json'); job=str(launch['job_id'])
    receipt_path=ROOT/'logs'/f'curriculum-targeted-case-assembly-v1-exit-{job}.json'
    receipt=read(receipt_path); observed=scheduler(job)
    if observed.returncode:
        if archive is None: raise ValueError('Actual full-shard assembler terminal is required')
        saved=read(archive)
        if saved['exit_receipt'] != receipt: raise ValueError('Actual assembly receipt differs')
        raw=saved['raw']
    else: raw=observed.stdout
    proof=terminal_record(raw,job,ENTRY)
    if (str(receipt['job_id']) != job or fields(raw)['NodeList'] != launch['node'].split('.')[0]
            or proof['successful'] and receipt['exit_code'] != 0):
        raise ValueError('Actual assembler job/node/exit required')
    return dict(**proof,exit_receipt=receipt,receipt_path=str(receipt_path))


def actual_dispatch():
    worker.verify_binding(); value=read(worker.DISPATCH)
    reg=read(original.RUN/'registered-private.json'); selected=reg['selected'][81:108]
    if (value['status'] != 'actual_ended_source4_missing_cases_frozen_for_disjoint_workers'
            or value['collection_sha256'] != sha(original.RUN/'registered-private.json')
            or value['worker_binding_sha256'] != sha(worker.BINDING)
            or value['numeric_producer_binding_sha256'] != sha(cached.BINDING)
            or value['shard'] != 3 or value['selected'] != selected
            or value['assignments'] != worker.partition.partition_missing(selected,value['completed_source_ids'],3)
            or any(sha(path) != expected for path,expected in value['files'].items())):
        raise ValueError('Exact actual ended27-view dispatch and case assignments required')
    return value,reg,selected


def lane_inputs(folder,archive,lane,assigned,reg):
    folder=Path(folder).resolve(); proof=worker.require_terminal(folder,archive)
    complete_path=folder/'partition-complete-private.json'; complete=read(complete_path)
    launch=read(folder/'registered-private.json'); summary=read(folder/'summary-private.json')
    if (not proof['successful'] or complete['status'] != worker.STATUS
            or complete['lane'] != lane or complete['assigned'] != assigned
            or complete['worker_binding_sha256'] != sha(worker.BINDING)
            or complete['dispatch_sha256'] != sha(worker.DISPATCH)
            or complete['original_numeric_producer_binding_sha256'] != sha(cached.BINDING)
            or not complete['exact_original_case_body_unchanged'] or complete['partial_world_targets_admitted']
            or summary['status'] != worker.STATUS or summary['selected_views'] != len(assigned)
            or launch['lane'] != lane or launch['selected'] != assigned):
        raise ValueError('Only actual complete successful same-assignment workers may be assembled')
    admission_path=Path(complete['admission']); admission=read(admission_path)
    files=dict(admission['files'])
    if any(sha(path) != expected for path,expected in files.items()):
        raise ValueError('Actual ended case-worker ancestry changed')
    for terminal in admission['terminals']:
        command=Path(fields(terminal['raw'])['Command'])
        if command not in (original.ROOT/'cluster/curriculum-targeted-win-labels-v1.slurm',cached.ENTRY,worker.ENTRY):
            raise ValueError('Unknown actual parent producer')
        actual=terminal_record(terminal['raw'],terminal['job_id'],command)
        if any(terminal[key] != expected for key,expected in actual.items()):
            raise ValueError('A preserved actual parent terminal differs')
        receipt_path=Path(terminal['receipt_path'])
        if read(receipt_path) != terminal['exit_receipt']:
            raise ValueError('Actual ended parent exit receipt changed')
        files[str(receipt_path)]=sha(receipt_path)
    cache_path=Path(complete['world_cache_manifest'])
    if sha(cache_path) != complete['world_cache_manifest_sha256']:
        raise ValueError('Actual complete-world cache manifest changed')
    cache=read(cache_path)
    expected_identity=dict(collection_sha256=sha(original.RUN/'registered-private.json'),
        teacher_identity=reg['continuation_model']['identity_sha256'],
        producer_binding_sha256=sha(cached.BINDING),purpose='actual_fixed_label_collection',
        world_cache_method=worker.METHOD)
    if (cache['method'] != worker.METHOD or cache['partial_world_targets_admitted']
            or not cache['all_current_arms_required'] or cache['original_collection_identity'] != expected_identity):
        raise ValueError('Only original complete-current-menu cache packets may be used')
    worker.combine_files(files,cache['files'])
    for name,expected in summary['files'].items():
        path=(folder/name).resolve()
        if sha(path) != expected: raise ValueError('Actual completed lane labels changed')
        files[str(path)]=expected
    for path in (complete_path,admission_path,cache_path,folder/'registered-private.json',folder/'summary-private.json'):
        files[str(path.resolve())]=sha(path)
    incoming,bound=original.completed_from(folder,assigned,reg)
    worker.combine_files(files,bound)
    if set(incoming) != {item['id'] for item in assigned}:
        raise ValueError('Every fixed assigned case must be complete or explicitly quarantined')
    if any(sha(path) != expected for path,expected in files.items()):
        raise ValueError('Actual completed lane input changed')
    return incoming,files,proof


def assemble(folders,archives):
    allocation=resource(); started=time.monotonic(); dispatch,reg,selected=actual_dispatch()
    if FULL.exists(): raise ValueError('Never overwrite complete or partial full-shard assembly')
    if len(folders) != len(dispatch['assignments']) or len(archives) != len(folders):
        raise ValueError('One actual completed final folder and archive per assigned lane required')
    recovered={}; files={}; common_terminals=[]; lane_terminals=[]
    worker.combine_files(files,read(worker.BINDING)['files'])
    for name,archive in zip(dispatch['original_priors'],dispatch['original_archives']):
        folder=Path(name); common_terminals.append(cached.require_terminal(folder,Path(archive)))
        incoming,bound=original.completed_from(folder,selected,reg)
        original.combine(recovered,incoming); worker.combine_files(files,bound)
    if set(recovered) != set(dispatch['completed_source_ids']):
        raise ValueError('Original ended source population differs from dispatch')
    for lane,(folder,archive) in enumerate(zip(folders,archives)):
        assigned=worker.partition.select_specs(selected,dispatch['assignments'][lane])
        incoming,bound,terminal=lane_inputs(folder,archive,lane,assigned,reg)
        if set(recovered)&set(incoming): raise ValueError('Parallel case assignments overlap')
        original.combine(recovered,incoming); worker.combine_files(files,bound); lane_terminals.append(terminal)
    if set(recovered) != {item['id'] for item in selected}:
        raise ValueError('All original27 views required, never a partial lane subset')
    FULL.mkdir(); (FULL/'cases-private').mkdir()
    write(FULL/'registered-private.json',dict(parent_registration_sha256=sha(original.RUN/'registered-private.json'),
        shard=3,selected=selected,model=reg['continuation_model'],job_id=os.environ['SLURM_JOB_ID'],
        node=socket.gethostname(),parallel_case_assembly=True,dispatch_sha256=sha(worker.DISPATCH)))
    counts=Counter(); contexts={}
    with (FULL/'labels-private.jsonl').open('x') as labels,(FULL/'quarantine-private.jsonl').open('x') as quarantine:
        for index,spec in enumerate(selected):
            original.reuse(recovered[spec['id']],spec,index,FULL/'cases-private',labels,quarantine,counts,contexts)
    if counts['labelled_views']+counts['sampling_quarantine'] != 27:
        raise ValueError('Original complete population accounting failed')
    summary=dict(status='fixed_targeted_win_label_shard_complete_candidate_targets_not_fit_admitted',
        shard=3,selected_views=27,counts=dict(counts),contexts={key:dict(value) for key,value in contexts.items()},
        files={name:sha(FULL/name) for name in ('labels-private.jsonl','quarantine-private.jsonl')},
        actual_seconds=time.monotonic()-started,teacher_model=reg['continuation_model']['identity_sha256'],
        uncertainty_retained=True,last_place_rate_veto=False,positive_confidence_gate=False,
        student_updates=0,job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),
        assembly_only_no_new_model_queries_or_rollouts=True)
    write(FULL/'summary-private.json',summary)
    for path in (worker.BINDING,worker.DISPATCH,worker.RUN/'queue-private.json',Path(__file__),FULL/'registered-private.json',FULL/'summary-private.json',
                 FULL/'labels-private.jsonl',FULL/'quarantine-private.jsonl'):
        files[str(path.resolve())]=sha(path)
    if any(sha(path) != expected for path,expected in files.items()):
        raise ValueError('Actual original source changed during full assembly')
    write(FULL/'parallel-assembly-proof-private.json',dict(status=STATUS,files=files,
        selected=selected,common_terminals=common_terminals,lane_terminals=lane_terminals,
        original_numeric_producer_binding_sha256=sha(cached.BINDING),
        all27_complete_cases_or_original_quarantine=True,all_original_label_objects_preserved=True,
        partial_world_targets_admitted=False,new_games=0,model_loaded=False,student_updates=0,allocation=allocation))
    print(json.dumps(dict(status=STATUS,selected_views=27,counts=dict(counts),new_games=0)),flush=True)


def verify_assembly(folder):
    folder=Path(folder).resolve()
    if folder != FULL.resolve(): raise ValueError('Only the explicit complete parallel source4 assembly is allowed')
    actual_dispatch(); proof=read(FULL/'parallel-assembly-proof-private.json')
    selected=read(original.RUN/'registered-private.json')['selected'][81:108]
    if (proof['status'] != STATUS or proof['selected'] != selected
            or not proof['all27_complete_cases_or_original_quarantine']
            or not proof['all_original_label_objects_preserved'] or proof['partial_world_targets_admitted']
            or proof['original_numeric_producer_binding_sha256'] != sha(cached.BINDING)
            or any(sha(path) != expected for path,expected in proof['files'].items())):
        raise ValueError('Actual complete successful parallel assembly provenance required')
    return {**proof['files'],str((FULL/'parallel-assembly-proof-private.json').resolve()):sha(FULL/'parallel-assembly-proof-private.json')}


def merge(folders,archives):
    """Reuse the original full108 merge, changing only actual lineage handling."""
    resource(); worker.verify_binding()
    base=pipeline.pipeline_function('merge')
    def terminal(folder,archive=None):
        return require_terminal(archive) if Path(folder).resolve()==FULL.resolve() else cached.require_terminal(folder,archive)
    def lineage(folder):
        return verify_assembly(folder) if Path(folder).resolve()==FULL.resolve() else pipeline.verify_recovery_lineage(folder)
    scope=dict(base.__globals__,require_terminal=terminal,verify_recovery_lineage=lineage)
    FunctionType(base.__code__,scope,base.__name__,base.__defaults__,base.__closure__)(folders,archives)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--phase',choices=('assemble','merge'),required=True)
    p.add_argument('--folder',type=Path,action='append',default=[])
    p.add_argument('--archive',type=Path,action='append',default=[])
    args=p.parse_args(); (assemble if args.phase=='assemble' else merge)(args.folder,args.archive)
