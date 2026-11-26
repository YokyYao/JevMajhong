"""Execute disjoint fixed cases using the original cached numeric producer.

No dispatch/entry binding is created here. A finite scheduled worker requires
actual ended common parents and a separately frozen complete-case assignment.
Partial worker output is never a full27-view shard or training admission.
"""
import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools import curriculum_targeted_case_partition_v1 as partition
from tools import curriculum_targeted_win_resume_v2 as previous
from tools import curriculum_targeted_win_resume_v3 as cached
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, terminal_record, fields
from tools.curriculum_targeted_world_cache_v1 import CompleteWorldCache, METHOD
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write

RUN = ROOT/'runs/curriculum-targeted-case-parallel-v1'
BINDING = RUN/'entry-binding-private.json'
DISPATCH = RUN/'dispatch-private.json'
ENTRY = ROOT/'cluster/curriculum-targeted-case-worker-v1.slurm'
EXIT_PREFIX = 'curriculum-targeted-case-worker-v1-exit-'
STATUS = 'fixed_targeted_disjoint_case_worker_complete_not_full_shard'


def verify_binding():
    value = read(BINDING)
    if (value['status'] != 'same_numeric_producer_parallel_case_entry_bound_not_dispatched'
            or any(sha(path) != expected for path,expected in value['files'].items())):
        raise ValueError('Before-dispatch numeric producer and worker source binding required')
    return value


def require_terminal(folder,archive=None):
    folder = Path(folder).resolve()
    if RUN not in folder.parents:
        return cached.require_terminal(folder,archive)
    launch = read(folder/'registered-private.json'); job = str(launch['job_id'])
    receipt_path = ROOT/'logs'/f'{EXIT_PREFIX}{job}.json'; receipt = read(receipt_path)
    actual = scheduler(job)
    if actual.returncode:
        if archive is None: raise ValueError('Actual case-worker terminal required; never infer it')
        saved = read(archive)
        if saved['exit_receipt'] != receipt: raise ValueError('Archived actual worker exit differs')
        raw = saved['raw']
    else: raw = actual.stdout
    proof = terminal_record(raw,job,ENTRY)
    if (str(receipt['job_id']) != job or fields(raw)['NodeList'] != launch['node'].split('.')[0]
            or proof['successful'] and receipt['exit_code'] != 0):
        raise ValueError('Actual worker job, node and exit receipt required')
    return dict(**proof,exit_receipt=receipt,receipt_path=str(receipt_path))


def dispatch(lane):
    verify_binding(); value = read(DISPATCH)
    if (value['status'] != 'actual_ended_source4_missing_cases_frozen_for_disjoint_workers'
            or value['collection_sha256'] != sha(previous.RUN/'registered-private.json')
            or value['worker_binding_sha256'] != sha(BINDING)
            or value['numeric_producer_binding_sha256'] != sha(cached.BINDING)
            or any(sha(path) != expected for path,expected in value['files'].items())):
        raise ValueError('Actual ended-source dispatch and original fixed numeric recipe required')
    reg = read(previous.RUN/'registered-private.json'); shard = value['shard']
    if shard != 3: raise ValueError('Only the actual slow source4 shard is partitioned')
    full = reg['selected'][shard*27:(shard+1)*27]
    if full != value['selected']:
        raise ValueError('Exact original full27 source objects required')
    expected = partition.partition_missing(full,value['completed_source_ids'],3)
    if value['assignments'] != expected or not 0 <= lane < len(expected):
        raise ValueError('Exact outcome-independent original-order case assignment required')
    return value,reg,full,partition.select_specs(full,expected[lane])


def combine_files(files,incoming):
    for name,expected in incoming.items():
        name = str(Path(name).resolve())
        if name in files and files[name] != expected:
            raise ValueError('Conflicting complete-case input identity')
        files[name] = expected


def run(lane,attempt,priors,archives):
    allocation = resource(); spec,reg,full,assigned = dispatch(lane)
    if type(attempt) is not int or attempt < 1 or len(priors) != len(archives):
        raise ValueError('Explicit bounded attempt and paired prior archives required')
    source = read(reg['source_registration'])
    sys.path[:0] = [source['source'],source['training_config']['upstream_source']]
    common = {}; files = {}; terminals = []
    common_priors = [Path(path).resolve() for path in spec['original_priors']]
    if len(common_priors) != len(spec['original_archives']):
        raise ValueError('Every actual ended common parent needs its real terminal')
    for folder,archive in zip(common_priors,spec['original_archives']):
        terminals.append(cached.require_terminal(folder,Path(archive)))
        incoming,bound = previous.completed_from(folder,full,reg)
        previous.combine(common,incoming); combine_files(files,bound)
    if set(common) != set(spec['completed_source_ids']):
        raise ValueError('Common ended complete-case population differs from the frozen assignment')
    # No common completed case is assigned to a worker. Hold only receipts,
    # not every old raw world packet, while loading the fixed teacher.
    del common; gc.collect()
    recovered = {}; prior_folders = [Path(path).resolve() for path in priors]
    lane_root = RUN/f'lane-{lane:02d}'
    if (len(set(prior_folders)) != len(prior_folders)
            or any(folder.parent != lane_root for folder in prior_folders)
            or len(prior_folders) != attempt-1):
        raise ValueError('All and only the actual earlier same-lane attempts are required')
    for index,(folder,archive) in enumerate(zip(prior_folders,archives),1):
        if folder.name != f'attempt-{index:03d}':
            raise ValueError('Missing or reordered actual same-lane recovery attempt')
        terminals.append(require_terminal(folder,archive))
        incoming,bound = previous.completed_from(folder,assigned,reg)
        previous.combine(recovered,incoming); combine_files(files,bound)
    if not set(recovered) <= {item['id'] for item in assigned}:
        raise ValueError('A recovered case is outside this lane')
    files[str(DISPATCH)] = sha(DISPATCH); files[str(BINDING)] = sha(BINDING)
    lane_root.mkdir(parents=True,exist_ok=True)
    out = lane_root/f'attempt-{attempt:03d}'
    admission = lane_root/f'attempt-{attempt:03d}-admission-private.json'
    write(admission,dict(status='actual_disjoint_case_parents_preserved_before_worker',
        lane=lane,attempt=attempt,assigned=assigned,terminals=terminals,files=files,
        original_priors=[str(path) for path in common_priors],
        own_priors=[str(path) for path in prior_folders],
        recovered_owned_cases=len(recovered),model_loaded=False,allocation=allocation))
    from tools.curriculum_endgame_ledger import ledger
    cache_identity = dict(collection_sha256=sha(previous.RUN/'registered-private.json'),
        teacher_identity=reg['continuation_model']['identity_sha256'],
        producer_binding_sha256=sha(cached.BINDING),purpose='actual_fixed_label_collection',
        world_cache_method=METHOD)
    # The key identifies the unchanged numeric producer. This new executor's
    # distinct actual entry/source binding is also retained in the completion.
    # Only readonly ended-parent caches are used; no live shared writers.
    old_caches = [path/'world-cache-private' for path in (*common_priors,*prior_folders)
        if (path/'world-cache-private').is_dir()]
    cache = CompleteWorldCache(out/'world-cache-private',old_caches,cache_identity,ledger)
    from tools import collect_curriculum_targeted_win_labels_v1 as collector
    def writer(path,value):
        if Path(path).name == 'summary-private.json':
            value = dict(value,status=STATUS,full27_shard_complete=False,
                lane=lane,partition_binding_sha256=sha(BINDING),
                original_numeric_producer_binding_sha256=sha(cached.BINDING))
        elif Path(path).name == 'registered-private.json':
            value = dict(value,lane=lane,attempt=attempt,original_parent_selected=full,
                actual_parallel_executor_binding_sha256=sha(BINDING))
        write(path,value)
    scope = dict(vars(collector),RESUME_OUT=out,RECOVERED=recovered,reuse=previous.reuse,
        world_cache=cache,select_specs=partition.select_specs,
        ASSIGNED_CASE_IDS=[item['id'] for item in assigned],write=writer)
    tree = partition.collector_tree(previous.COLLECTOR.read_text())
    exec(compile(tree,str(previous.COLLECTOR),'exec'),scope); scope['collect'](spec['shard'])
    if any(sha(path) != expected for path,expected in files.items()):
        raise ValueError('An actual ended parent changed during case collection')
    manifest = cache.seal(out/'world-cache-binding-private.json')
    write(out/'partition-complete-private.json',dict(
        status=STATUS,lane=lane,attempt=attempt,assigned=assigned,admission=str(admission),
        worker_binding_sha256=sha(BINDING),dispatch_sha256=sha(DISPATCH),
        original_numeric_producer_binding_sha256=sha(cached.BINDING),
        exact_original_case_body_unchanged=True,world_cache_method=METHOD,
        world_cache_manifest=str(manifest),world_cache_manifest_sha256=sha(manifest),
        actual_owned_cases=len(assigned),recovered_owned_cases=len(recovered),
        full27_shard_not_admitted=True,partial_world_targets_admitted=False,student_updates=0))
    print(json.dumps(dict(status=STATUS,lane=lane,assigned=len(assigned),
        full27_shard_not_admitted=True,student_updates=0)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lane',type=int,required=True); p.add_argument('--attempt',type=int,required=True)
    p.add_argument('--prior',type=Path,action='append',default=[])
    p.add_argument('--archive',type=Path,action='append',default=[])
    args=p.parse_args(); run(args.lane,args.attempt,args.prior,args.archive)
