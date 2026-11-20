"""Trace existing MC rows back to private rollout artifacts without relabeling."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
FIELDS=('seat','hand','rivers','melds','dora_indicators','discards_count','honba','sticks')


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def position(row,group):
    payload={k:row['state'].get(k) for k in FIELDS}
    return group+'|'+hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load_rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def content_identity(row):
    # Public exports hash private IDs/groups. Match the complete model input
    # and target, never infer a native seed from an anonymized group string.
    return hashlib.sha256(json.dumps([row['state'],row['questions']],sort_keys=True,separators=(',',':')).encode()).hexdigest()


def trace(data_ready,out,probe_shards=3):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-','login'))
    data=read(data_ready);wanted={};membership={};phase_counts=Counter()
    for split,spec in data['files'].items():
        file=ROOT/spec['path'];assert sha(file)==spec['sha256']
        for row in load_rows(file):
            if row['_meta']['label_type']!='paired_future_masked':continue
            identity=row['_meta']['id'];assert identity not in wanted
            wanted[identity]=row;membership[identity]=split;phase_counts[row['_meta']['phase']]+=1
    original=ROOT/'runs/unified-attackrisk-data-v8';original_ready=read(original/'result.json');old={};old_split={}
    for split,spec in original_ready['files'].items():
        assert sha(spec['path'])==spec['sha256']
        for row in load_rows(spec['path']):
            if row['_meta']['label_type']=='paired_future_masked':
                assert row['_meta']['id'] not in old
                old[row['_meta']['id']]=row;old_split[row['_meta']['id']]=split
    indexed={}
    for identity,row in old.items():
        key=content_identity(row);assert key not in indexed,'Ambiguous original contents'
        indexed[key]=identity
    links={}
    for identity,row in wanted.items():
        key=content_identity(row);assert key in indexed,'Unmatched public model input/target'
        links[identity]=indexed[key]
    assert len(set(links.values()))==len(wanted)==len(old),'Incomplete/duplicated provenance map'
    native_groups={old[links[k]]['_meta']['group_id'] for k in wanted}
    seed_groups={int(group.split(':')[1]) for group in native_groups if group.startswith('native:') and group.split(':')[1].isdigit()}
    assert len(seed_groups)==len(native_groups)
    assert not seed_groups&set(range(2026300000,2026300032))
    byposition=defaultdict(list)
    for identity,row in wanted.items():byposition[position(row,old[links[identity]]['_meta']['group_id'])].append(identity)
    inputs_path=original/'inputs-private.json';inputs=read(inputs_path)
    families=defaultdict(list)
    for path,h in inputs.items():
        p=Path(path)
        if p.suffix=='.pt' and p.parent.name=='walls-private':families[p.parent.parent].append((p,h))
    metadata=[];eligible=[]
    for directory,files in sorted(families.items()):
        protocol=directory/'protocol.json'
        if not protocol.exists():continue
        payload=read(protocol)
        if payload.get('collection_method')!='paired_future_ev_v1':continue
        cfg=payload['config'];relevant=[(p,h) for p,h in files if p.stem.isdigit() and int(p.stem) in seed_groups]
        exists=[p for p,_ in relevant if p.exists()]
        source_files={p:h for p,h in payload.get('code_sha256',{}).items() if p.endswith('/training/futures.py')}
        verified_sources={p:h for p,h in source_files.items() if Path(p).exists() and sha(p)==h}
        metadata.append(dict(directory=str(directory.relative_to(ROOT)),protocol_sha256=sha(protocol),
            protocol_bound=inputs.get(str(protocol))==sha(protocol),bound_shards=len(files),relevant_shards=len(relevant),
            present_relevant_shards=len(exists),present_bytes=sum(p.stat().st_size for p in exists),
            worlds=cfg.get('worlds'),wall_scope=cfg.get('wall_scope'),reward=cfg.get('reward'),
            decision_filter=cfg.get('decision_filter'),producer_futures_sources=verified_sources,
            continuation={k:cfg.get(k) for k in ('discard_checkpoint','call_checkpoint','call_context_baseline')},
            stop_horizon_requires_producer_source_check=True))
        if relevant:eligible.append((directory,relevant,payload))
    import torch
    torch.set_num_threads(1)
    probes=[]
    for directory,relevant,protocol in sorted(eligible,key=lambda v:(-len(v[1]),str(v[0])))[:probe_shards]:
        path,expected=next((p,h) for p,h in relevant if p.exists());assert sha(path)==expected,path
        payload=torch.load(path,map_location='cpu',weights_only=False);windows=defaultdict(list)
        for row in payload.get('rows',[]):
            if 'counterfactual_worlds' not in row:continue
            group='native:'+str(row['episode_key'][0]);pos=position(row['public_request'],group)
            if pos in byposition:windows[pos].append(row)
        world_counts=Counter();current_matches=0;consistent=0
        for pos,group_rows in windows.items():
            current_matches+=len(byposition[pos]);counts={len(r['counterfactual_worlds']) for r in group_rows}
            assert len(counts)==1;count=counts.pop();world_counts[count]+=1
            same_seeds=all([w['future_seed'] for w in r['counterfactual_worlds']]==
                [w['future_seed'] for w in group_rows[0]['counterfactual_worlds']] for r in group_rows)
            consistent+=same_seeds and count==protocol['config']['worlds']
        probes.append(dict(directory=str(directory.relative_to(ROOT)),shard_sha256=expected,sha_verified=True,
            shard_rows=len(payload.get('rows',[])),matching_positions=len(windows),matching_current_records=current_matches,
            actual_world_counts=dict(world_counts),aligned_world_count_and_seed_positions=consistent,
            complete_target_reproduction_not_yet_checked=True))
        del payload
    result=dict(status='private_mc_source_trace_complete_not_target_admission',node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
        current_mc_records=len(wanted),current_split_counts=dict(Counter(membership.values())),current_phases=dict(phase_counts),
        native_source_groups=len(native_groups),numeric_native_seeds=len(seed_groups),reserved_raw_seed_overlap=0,
        original_v8_mc_records=len(old),exact_original_ID_matches=len(set(wanted)&set(old)),
        exact_original_content_matches=len(links),all_anonymized_rows_have_unique_private_source=True,
        original_v8_split_counts=dict(Counter(old_split.values())),
        original_data_receipt_sha256=sha(original/'result.json'),original_input_manifest_sha256=sha(inputs_path),
        future_source_families=metadata,probes=probes,
        original_train_rows_now_held=sum(old_split[links[k]]=='train' and membership[k].startswith('validation') for k in wanted),
        model_loaded=False,targets_changed=False,optimizer_updates=0,current_fits_modified=False,
        confidence_or_complete_label_provenance_not_yet_reconstructed=True,
        note='Unique complete input/target content links recover anonymized IDs. Sampled aligned worlds are not confidence labels or target admission. Verify each bound producer sampler and stop horizon: inspected producer46d keeps actual opponent hands fixed and shuffles only wall scope. This is conditional rollout, not a public-information belief. Full row selection priority, reward/bootstrap reproduction and held-out exclusions remain required. No raw records are published.')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('future_source_families',)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-ready',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--probe-shards',type=int,default=3)
    a=p.parse_args();trace(a.data_ready,a.out,a.probe_shards)
