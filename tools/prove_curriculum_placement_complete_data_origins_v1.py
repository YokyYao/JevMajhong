"""Prove every complete placement row and virtual copy against actual parents.

This is dataset-origin evidence, not source-seed closure or candidate admission.
It runs only in finite scheduler GPU resources and never loads model weights.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_confirmation_json_sources_v1 import _object,_constant
from tools.curriculum_placement_exposures_v1 import REPEAT_FIELD

EXPOSURES=ROOT/'runs/curriculum-placement-recovered-exposures-v1/data-v1/data-ready-private.json'
COHERENT=ROOT/'runs/curriculum-targeted-win-resume-v3/placement-data-v5/data-v1/data-ready-private.json'
TARGETED=ROOT/'runs/curriculum-targeted-win-resume-v3/assembled-v1/merged-v1/data-ready-private.json'
BASE=ROOT/'runs/curriculum-placement-data-v2/data-v1/data-ready-private.json'
OUT=ROOT/'runs/curriculum-placement-complete-data-origin-proof-v1'


def row_sha(row):
    # Preserve object insertion order, including the model's candidate order.
    return hashlib.sha256(json.dumps(row,ensure_ascii=False,allow_nan=False,
        separators=(',',':')).encode()).hexdigest()


def restore_copy(row,repeats):
    meta=row['_meta']; trace=meta.get(REPEAT_FIELD)
    if trace is None: return row,meta['id'],0
    source=row['questions']['action']['src']; expected=repeats[source]
    if (set(trace) != {'original_id','copy_index','total_exposures_per_pass','adds_independent_data'}
            or trace['adds_independent_data'] is not False
            or type(trace['copy_index']) is not int or not 1 <= trace['copy_index'] < expected
            or trace['total_exposures_per_pass'] != expected):
        raise ValueError('Only the original bounded explicit virtual-copy trace is allowed')
    rid=trace['original_id']
    expected_id='placement-repeat-v1:'+hashlib.sha256(rid.encode()).hexdigest()+':'+str(trace['copy_index'])
    if meta['id'] != expected_id: raise ValueError('Virtual ID differs from the original exposure recipe')
    restored=dict(row,_meta={key:value for key,value in meta.items() if key != REPEAT_FIELD})
    restored['_meta']['id']=rid
    return restored,rid,trace['copy_index']


def require_origin(row,split,catalogue):
    rid=row['_meta']['id']; expected=catalogue.get(rid)
    if (expected is None or expected['split'] != split
            or expected['sha256'] != row_sha(row)
            or expected['group'] != row['_meta']['group_id']
            or expected['source'] != row['questions']['action']['src']):
        raise ValueError('A row changed its parent, current file split, ordered input, target or source')
    return expected


def prove():
    allocation=resource(); started=time.monotonic()
    # The genuine complete exposure output is required before any proof writes.
    expanded=read(EXPOSURES)
    if OUT.exists(): raise ValueError('Never overwrite complete or partial dataset-origin evidence')
    coherent=read(COHERENT); targeted=read(TARGETED); base=read(BASE)
    if (expanded['status'] != 'placement_expanded_corpus_and_two_epoch_recipe_prepared_not_parent_or_fit_registered'
            or coherent['status'] != 'placement_all_sources_including_targeted_win_special_prepared_not_fit'
            or targeted['selected_views'] != 108 or not targeted['all_original_families_and_partitions_preserved']
            or not expanded['originals_preserved_object_exact']
            or not expanded['replicas_change_only_ID_and_explicit_exposure_trace']):
        raise ValueError('Actual complete original data and exposure preparation are required')
    files={str(Path(__file__).resolve()):sha(__file__)}; metadata={}; dataset_stats={}
    for ready in (EXPOSURES,COHERENT,TARGETED,BASE):
        receipt_path=ready.parent/'completion-binding-private.json'; receipt=read(receipt_path)
        metadata[str(ready)]=sha(ready); metadata[str(receipt_path)]=sha(receipt_path)
        # Reuse existing completed prerequisite identities; raw data is hashed
        # once in the stream below rather than rescanned for numeric seeds.
        for name,expected in receipt['files'].items():
            key=str((ROOT/name).resolve())
            if key in files and files[key] != expected: raise ValueError('Conflicting completed source identity')
            files[key]=expected
    source_path=ROOT/'runs/curriculum-progression-v1/fit-plan/registered.json'
    source=read(source_path)
    if files.get(str(source_path)) != sha(source_path): raise ValueError('Actual frozen projection source registration required')
    sys.path[:0]=[source['source'],source['training_config']['upstream_source']]
    from tools.curriculum_placement_input_v1 import training_record
    from jevmahjong.policy.unified import serving_request as normalizer
    normalizer_path=Path(sys.modules['jevmahjong.policy.unified'].__file__).resolve()
    if normalizer_path != (Path(source['source'])/'jevmahjong/policy/unified.py').resolve():
        raise ValueError('Actual frozen serving normalizer required')
    base_receipt_path=ROOT/'runs/curriculum-placement-base-source-closure-v1/completion-binding-private.json'
    base_receipt=read(base_receipt_path)['files']
    if base_receipt.get(str(normalizer_path)) != sha(normalizer_path):
        raise ValueError('Completed base normalizer identity differs')
    metadata[str(base_receipt_path)]=sha(base_receipt_path)
    files[str(normalizer_path)]=sha(normalizer_path)
    projection_path=ROOT/'tools/curriculum_placement_input_v1.py'
    if files.get(str(projection_path)) != sha(projection_path):
        raise ValueError('Actual prepared projection implementation changed')

    def stream(spec):
        path=(ROOT/spec['path']).resolve(); before=path.stat(); digest=hashlib.sha256(); count=0; length=0
        if str(path) in files and files[str(path)] != spec['sha256']:
            raise ValueError('Dataset specification conflicts with its actual completed producer binding')
        with path.open('rb') as handle:
            for raw in handle:
                digest.update(raw); length+=len(raw)
                if not raw.strip(): continue
                if not raw.endswith(b'\n'): raise ValueError('Complete ended JSONL records required')
                count+=1
                yield json.loads(raw,object_pairs_hook=_object,parse_constant=_constant)
        after=path.stat()
        if (digest.hexdigest() != spec['sha256'] or count != spec['records']
                or length != before.st_size or length != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns):
            raise ValueError('Actual complete dataset identity/count changed during the stream read')
        files[str(path)]=spec['sha256']; dataset_stats[str(path)]=(after.st_size,after.st_mtime_ns)

    catalogue={}; groups=defaultdict(set); original_pools=Counter()
    for split in ('train','validation'):
        for label,parent in (('closed_base',base),('complete_targeted',targeted)):
            for row in stream(parent['files'][split]):
                before=row_sha(row); projected=training_record(row,normalizer=normalizer)
                if row_sha(row) != before: raise ValueError('Projection mutated an original parent')
                if label=='closed_base' and projected != row:
                    raise ValueError('The coherent completed base changed on projection')
                for field in ('label','target','src','criteria'):
                    if projected['questions']['action'].get(field) != row['questions']['action'].get(field):
                        raise ValueError('Parent supervision/menu changed on projection')
                meta=projected['_meta']; rid=meta['id']; source_id=projected['questions']['action']['src']
                if rid in catalogue or REPEAT_FIELD in meta:
                    raise ValueError('Original parents must have distinct pristine IDs')
                catalogue[rid]=dict(split=split,sha256=row_sha(projected),group=meta['group_id'],source=source_id,pool=label)
                groups[split].add(meta['group_id']); original_pools[split+':'+label]+=1
    if groups['train']&groups['validation']: raise ValueError('Actual parent source families cross current file partitions')
    observed=set(); train_order=[]; source_counts={}; coverage=[]
    for split in ('train','validation'):
        counts=Counter(); chain=hashlib.sha256(); records=0
        for row in stream(coherent['files'][split]):
            rid=row['_meta']['id']; require_origin(row,split,catalogue)
            if rid in observed: raise ValueError('Original source ID duplicated across complete current files')
            observed.add(rid); records+=1; counts[row['questions']['action']['src']]+=1
            chain.update(row_sha(row).encode())
            if split=='train': train_order.append(rid)
        if dict(counts) != coherent['source_records'][split]: raise ValueError('Actual full source counts differ')
        source_counts[split]=dict(counts)
        coverage.append(dict(dataset=str((ROOT/coherent['files'][split]['path']).resolve()),split=split,
            records=records,sha256=coherent['files'][split]['sha256'],ordered_row_hash_chain=chain.hexdigest(),
            every_parent_field_and_candidate_order_exact=True))
    if observed != set(catalogue): raise ValueError('A projected parent was omitted or an extra row was introduced')
    if expanded['files']['validation'] != coherent['files']['validation']:
        raise ValueError('Validation must be reused unchanged, never virtually repeated')
    plan_path=EXPOSURES.parent.parent/'proposal-v1/proposal-private.json'; plan=read(plan_path)
    metadata[str(plan_path)]=sha(plan_path)
    repeats=plan['virtual_repeats_per_original_by_source']; copies=defaultdict(set)
    seen=set(); expanded_counts=Counter(); chain=hashlib.sha256(); actual=0
    for index,row in enumerate(stream(expanded['files']['train'])):
        rid=row['_meta']['id']
        if rid in seen: raise ValueError('An expanded-row ID occurs twice')
        seen.add(rid); restored,origin,copy_index=restore_copy(row,repeats)
        require_origin(restored,'train',catalogue)
        if index<len(train_order):
            if rid != train_order[index] or copy_index != 0: raise ValueError('The original TRAIN prefix changed')
        elif copy_index == 0: raise ValueError('An appended exposure lacks its explicit virtual-copy trace')
        if copy_index in copies[origin]: raise ValueError('A copy index was duplicated')
        copies[origin].add(copy_index); actual+=1
        expanded_counts[row['questions']['action']['src']]+=1; chain.update(row_sha(row).encode())
    train_ids={rid for rid,item in catalogue.items() if item['split']=='train'}
    if set(copies)!=train_ids or any(indices != set(range(repeats[catalogue[rid]['source']])) for rid,indices in copies.items()):
        raise ValueError('Every original requires exactly its original bounded uniform exposure multiplicity')
    if (dict(expanded_counts) != expanded['source_exposures']
            or actual != expanded['training_exposures_per_pass']
            or len(train_ids) != expanded['unique_train_records']
            or len(train_ids) != coherent['files']['train']['records']):
        raise ValueError('Actual complete unique and expanded exposure counts differ')
    for name,(size,mtime) in dataset_stats.items():
        stat=Path(name).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(size,mtime): raise ValueError('A completed source changed after streaming')
    if any(sha(path)!=expected for path,expected in metadata.items()): raise ValueError('Completed metadata changed')
    if any(name in files and files[name] != expected for name,expected in metadata.items()):
        raise ValueError('Actual metadata conflicts with its completed upstream identity')
    files.update(metadata)
    OUT.mkdir(parents=True,exist_ok=False)
    write(OUT/'dataset-projection-private.json',coverage)
    write(OUT/'source-counts-private.json',source_counts)
    write(OUT/'repeated-original-coverage-private.json',{rid:sorted(indices) for rid,indices in copies.items() if len(indices)>1})
    summary=dict(status='actual_complete_placement_parent_projection_and_all_virtual_exposure_origins_proved',
        original_pools=dict(original_pools),unique_train_records=len(train_ids),
        unique_validation_records=len(catalogue)-len(train_ids),training_exposures_per_pass=actual,
        expanded_train_sha256=expanded['files']['train']['sha256'],expanded_ordered_row_hash_chain=chain.hexdigest(),
        loss_sources=len(expanded_counts),all_parent_rows_accounted_exactly_once=True,
        all_inputs_targets_sources_and_candidate_order_preserved=True,
        all_virtual_copies_change_only_original_id_and_explicit_trace=True,
        inherited_meta_split_not_reinterpreted_as_current_file_partition=True,
        validation_copies=0,unmatched_rows=0,cross_split_source_families=0,
        raw_world_numeric_and_seed_proofs_not_recomputed=True,
        complete_final_source_closure=False,actual_final_candidate_admitted=False,
        new_games=0,model_loaded=False,student_updates=0,allocation=allocation,seconds=time.monotonic()-started)
    write(OUT/'summary-private.json',summary)
    for path in OUT.glob('*-private.json'): files[str(path)]=sha(path)
    write(OUT/'completion-binding-private.json',dict(files=files,actual_final_candidate_admitted=False))
    print(json.dumps({key:value for key,value in summary.items() if key!='allocation'}),flush=True)


if __name__=='__main__': prove()
