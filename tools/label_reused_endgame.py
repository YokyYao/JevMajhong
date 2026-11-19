"""Freeze, distribute and reconcile original-v8 labels for reused sources."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys


def partition_rows(rows):
    """Balance public action counts; each original source belongs to one worker."""
    bins=[[],[]];loads=[0,0]
    for row in sorted(rows,key=lambda r:(-len(r['request']['questions']['action']['criteria']),r['seed'])):
        worker=min(range(2),key=lambda i:(loads[i],i))
        bins[worker].append(row);loads[worker]+=len(row['request']['questions']['action']['criteria'])
    return bins


def validate_terminal_menu(label,row,identity):
    from jevmahjong.endgame.context import placements,terminal_utility
    keys=list(label['record']['questions']['action']['criteria'])
    if (label['status']!='complete_terminal_menu' or label['source']!=row or label['worlds_per_action']!=32
            or label['source_request_sha256']!=row['request_sha256'] or not label['continued_until_native_match_done']
            or label['baseline_identity_sha256']!=identity or label['actual_actions']!=len(keys)
            or len(label['outcomes'])!=len(keys)):
        raise ValueError('Terminal menu provenance differs')
    for outcomes in label['outcomes']:
        if len(outcomes)!=32:raise ValueError('Incomplete paired worlds')
        for outcome in outcomes:
            if (not outcome['ledger_passed'] or placements(outcome['scores'])!=outcome['ranks']
                    or terminal_utility(outcome['ranks'][row['seat']])!=outcome['utility']):
                raise ValueError('Terminal ledger/rank/utility differs')
    return len(keys)*32


def frozen_protocol(config):
    from jevmahjong.endgame.collect import digest
    from jevmahjong.endgame.futures import bundle_identity
    root=Path(__file__).resolve().parents[1];out=Path(config['out'])
    if (config.get('schema')!='reused-fourth-label32-v1' or config.get('future_worlds')!=32
            or config.get('continuation_policy')!='released_original_v1' or config.get('label_registered') is not True
            or config.get('fit_registered') is not False or config.get('workers')!=2):
        raise ValueError('Only the registered two-worker original-v8 label stage is enabled')
    positions=json.loads((out/'selected-positions-private.json').read_text())
    if len(positions)!=80 or digest(positions)!=config['positions_sha256'] or len({r['seed'] for r in positions})!=80:
        raise ValueError('Fixed80 source manifest differs')
    paths=set()
    for directory in ('jevmahjong/core','jevmahjong/policy','jevmahjong/evaluation','desktop/jev_sidecar'):
        paths.update((root/directory).rglob('*.py'))
    paths.update(root/'jevmahjong/endgame'/name for name in (
        'collect.py','context.py','futures.py','ledger.py','native_public.py','policy.py','public_policy.py','win_value.py'))
    paths.update(root/name for name in ('jevmahjong/training/futures.py','tools/train_endgame.py',
                                       'tools/check_gpu.py','tools/label_reused_endgame.py'))
    # Offline prepare/fit/mix/reference modules are not called during labeling;
    # they can be developed independently without changing this frozen runtime.
    upstream=Path(config['upstream_source'])/'jevany'
    if not upstream.is_dir():raise ValueError('Pinned JevAny package missing')
    return {'schema':config['schema'],'configuration':config,'positions_sha256':digest(positions),
        'continuation_identity':bundle_identity(config['checkpoint']),
        'source_sha256':{str(r['seed']):hashlib.sha256((out/'sources'/f"{r['seed']}.json").read_bytes()).hexdigest() for r in positions},
        'public_context_sha256':hashlib.sha256((out/'public-context-private.json').read_bytes()).hexdigest(),
        'code_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
        'upstream_sha256':{str(p.relative_to(upstream)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(upstream.rglob('*.py'))},
        'package_versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','peft','riichienv','mahjong')},
        'worker_partitions':[[r['seed'] for r in rows] for rows in partition_rows(positions)],'optimizer_updates':0}


def prepare(config):
    from jevmahjong.endgame.collect import write,digest
    from desktop.jev_sidecar.unified_policy import identity
    out=Path(config['out']);protocol=frozen_protocol(config)
    if protocol['continuation_identity']['identity_sha256'] not in (identity()['source_identity_sha256'],identity()['identity_sha256']):
        raise ValueError('Continuation must be exactly the released original v8')
    positions=json.loads((out/'selected-positions-private.json').read_text())
    for row in positions:
        source=json.loads((out/'sources'/f"{row['seed']}.json").read_text())
        if 'hand_score_history' not in source:raise ValueError('Precomputed source ledger missing; workers cannot modify frozen sources')
    path=out/'label-protocol-private.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Existing frozen protocol differs')
    for index,rows in enumerate(partition_rows(positions)):
        child=out/'label-workers'/str(index);child.mkdir(parents=True,exist_ok=True)
        files={'selected-positions-private.json':rows,'config-private.json':dict(config,out=str(child))}
        for name,value in files.items():
            target=child/name
            if target.exists() and json.loads(target.read_text())!=value:raise ValueError('Existing worker input differs')
            write(target,value)
        link=child/'sources'
        if link.exists() and link.resolve()!=(out/'sources').resolve():raise ValueError('Worker source alias differs')
        if not link.exists():link.symlink_to((out/'sources').resolve(),target_is_directory=True)
    write(path,protocol)
    result={'status':'two_worker_labels_registered_and_frozen','positions':80,'future_worlds':32,
        'worker_positions':[len(x) for x in partition_rows(positions)],
        'worker_legal_actions':[sum(len(r['request']['questions']['action']['criteria']) for r in x) for x in partition_rows(positions)],
        'protocol_sha256':digest(protocol),'optimizer_updates':0}
    write(out/'label-preparation-private.json',result);return result


def worker(config,index):
    from jevmahjong.endgame.collect import write,digest
    from jevmahjong.endgame.futures import label
    if index not in (0,1):raise ValueError('Worker index must be0 or1')
    out=Path(config['out']);protocol=json.loads((out/'label-protocol-private.json').read_text())
    if protocol!=frozen_protocol(config):raise ValueError('Frozen label runtime changed before worker start')
    child=out/'label-workers'/str(index);rows=json.loads((child/'selected-positions-private.json').read_text())
    expected=partition_rows(json.loads((out/'selected-positions-private.json').read_text()))[index]
    child_config=json.loads((child/'config-private.json').read_text())
    if rows!=expected or child_config!=dict(config,out=str(child)):raise ValueError('Worker inputs differ')
    result=label(child_config)
    if protocol!=frozen_protocol(config):raise ValueError('Frozen label runtime changed during worker')
    write(child/'complete-private.json',{'status':'frozen_worker_complete','positions':result['positions'],
        'protocol_sha256':digest(protocol),'optimizer_updates':0});return result


def finalize(config):
    from jevmahjong.endgame.collect import digest,write
    from jevmahjong.endgame.context import placements,terminal_utility
    out=Path(config['out']);protocol=json.loads((out/'label-protocol-private.json').read_text())
    if protocol!=frozen_protocol(config):raise ValueError('Frozen protocol differs at final verification')
    positions=json.loads((out/'selected-positions-private.json').read_text())
    hashes={};world_rollouts=0
    for index,rows in enumerate(partition_rows(positions)):
        child=out/'label-workers'/str(index)
        receipt=json.loads((child/'complete-private.json').read_text())
        if receipt['protocol_sha256']!=digest(protocol) or receipt['positions']!=len(rows):
            raise ValueError('A worker is incomplete or has different provenance')
        for row in rows:
            rid=digest([row['seed'],row['step'],row['seat'],row['request_sha256']]);path=child/'labels'/(rid+'.json')
            label=json.loads(path.read_text())
            world_rollouts+=validate_terminal_menu(label,row,protocol['continuation_identity']['identity_sha256'])
            link=out/'labels'/(rid+'.json');link.parent.mkdir(parents=True,exist_ok=True)
            if link.exists() and link.resolve()!=path.resolve():raise ValueError('Original label alias differs')
            if not link.exists():link.symlink_to(path.resolve())
            hashes[rid]=hashlib.sha256(path.read_bytes()).hexdigest()
    result={'status':'frozen_label_protocol_verified','positions':len(positions),
        'positions_sha256':digest(positions),'source_groups':80,'future_worlds':32,
        'protocol_sha256':digest(protocol),'labels_sha256':hashes,'terminal_rollouts':world_rollouts,
        'continuation_policy':'released_original_v1','optimizer_updates':0}
    write(out/'label-protocol-verification-private.json',result)
    write(out/'label-result-private.json',{'status':'all_selected_terminal_labels_complete','positions':80,
        'worlds_per_action':32,'terminal_rollouts':world_rollouts,'optimizer_updates':0})
    return {k:v for k,v in result.items() if k!='labels_sha256'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('prepare','worker','finalize'))
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--worker-index',type=int)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Label work requires a scheduled GPU allocation, never login')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    config=json.loads(args.config.read_text());sys.path.insert(0,config['upstream_source'])
    result=prepare(config) if args.stage=='prepare' else worker(config,args.worker_index) if args.stage=='worker' else finalize(config)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
