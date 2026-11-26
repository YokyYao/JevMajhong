"""Cache only augmented TRAIN RI parent probabilities; no student updates."""
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_ri_parent_anchor_v1 import RI_SOURCES,make_targets

DATA=ROOT/'runs/curriculum-ri-augmented-data-v1/data-v1'
OUT=ROOT/'runs/curriculum-ri-augmented-data-v1/parent-targets-v1'
PROBE=ROOT/'runs/curriculum-ri-balanced618-full-TRAIN-probe-v1'


def main():
    allocation=resource();started=time.monotonic()
    if OUT.exists():raise ValueError('Never overwrite failed or completed parent target evidence')
    bindings={}
    for path in (DATA/'completion-binding-private.json',PROBE/'completion-binding-private.json'):
        for name,expected in read(path)['files'].items():
            if name in bindings and bindings[name]!=expected:raise ValueError('Conflicting prerequisite identity')
            bindings[name]=expected
        bindings[str(path)]=sha(path)
    if any(sha(path)!=expected for path,expected in bindings.items()):raise ValueError('Frozen sources changed')
    summary=read(DATA/'summary-private.json')
    if (summary['unique_rows']!={'train':92570,'validation':24998} or summary['RI_TRAIN_records']!=4244
            or summary['student_fit_admitted'] is not False):raise ValueError('Exact independently prepared standby union required')
    registered=ROOT/'runs/curriculum-ri-balanced-replay-fit-v2/fit-plan/registered.json';reg=read(registered)
    expected=reg['curriculum']['parent_identity'];old=read(PROBE/'summary-private.json')
    if old['actual_models']['0']!=expected or old['TRAIN_rows']!=2353:
        raise ValueError('Only exact qualified initialization probabilities may be reused')
    train=DATA/'train-union-private.jsonl';rows=[json.loads(line) for line in train.open()]
    selected=[r for r in rows if r['questions']['action']['src'] in RI_SOURCES]
    validation_ids=set()
    for values in read(DATA/'cohort-ids-private.json')['validation'].values():validation_ids.update(values)
    if len(rows)!=92570 or len(selected)!=4244 or any(r['_meta']['id'] in validation_ids for r in selected):
        raise ValueError('All and only4244 TRAIN RI menus, disjoint from full VAL IDs, required')
    old_path=PROBE/'step-000000-TRAIN-predictions-private.jsonl'
    reused=[json.loads(line) for line in old_path.open()];reused_ids={p['id'] for p in reused}
    if len(reused)!=2353 or len(reused_ids)!=2353:
        raise ValueError('Complete distinct2353 original TRAIN target cache required')
    original_ids=set(read(DATA/'cohort-ids-private.json')['train']['original'])
    if reused_ids!={r['_meta']['id'] for r in selected if r['_meta']['id'] in original_ids}:
        raise ValueError('Reused cache must equal all original RI TRAIN membership')
    fresh=[r for r in selected if r['_meta']['id'] not in reused_ids]
    if len(fresh)!=1891:raise ValueError('Exactly1891 new audited TRAIN RI requests required')
    OUT.mkdir()
    for path in (Path(__file__),registered,train,old_path,ROOT/'tools/curriculum_ri_parent_anchor_v1.py'):
        bindings[str(path)]=sha(path)
    write(OUT/'registered-private.json',dict(files=bindings,qualified_parent_identity=expected,
        all_selected_TRAIN_ids=[r['_meta']['id'] for r in selected],reused=2353,fresh=1891,
        selection_before_inference=True,no_VAL_or_test_requests_inferred=True,
        new_hard_labels_from_Mortal_retention_probabilities_from_qualified_student=True,
        GPU_memory_fraction=.25,student_updates=0,allocation=allocation))
    sys.path[:0]=[reg['source'],reg['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    if torch.cuda.device_count()!=1:raise ValueError('Exactly one assigned GPU required')
    if torch.cuda.mem_get_info()[0]<20*1024**3:raise RuntimeError('Insufficient free memory; defer rather than risk active fit')
    torch.cuda.set_per_process_memory_fraction(.25,0)
    import jevmahjong
    if not Path(jevmahjong.__file__).resolve().is_relative_to(Path(reg['source']).resolve()):
        raise ValueError('Exact pinned policy runtime required')
    from jevany.data import api_request
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    directory=Path(reg['parent_model']['directory']);identity=bundle(directory)
    if identity['identity_sha256']!=expected:raise ValueError('Wrong retention teacher model')
    loaded=load_model(directory);loaded.runtime.model.eval();client=UnifiedPolicyClient(loaded)
    destination=OUT/'parent-TRAIN-predictions-private.jsonl';predictions=list(reused)
    with destination.open('wb') as stream,torch.inference_mode():
        with old_path.open('rb') as source:
            for line in source:stream.write(line)
        for done,row in enumerate(fresh,1):
            q=row['questions']['action'];keys=list(q['criteria'])
            p=client.system_one(**api_request(row))['answers']['action']['probabilities']
            if (set(p)!=set(keys) or any(not math.isfinite(v) or v<0 for v in p.values())
                    or abs(sum(p.values())-1)>1e-5):raise ValueError('Complete finite actual menu probabilities required')
            result=dict(id=row['_meta']['id'],keys=keys,p=[p[k] for k in keys]);predictions.append(result)
            stream.write((json.dumps(result)+'\n').encode());stream.flush()
            if done%256==0:print(json.dumps(dict(stage='augmented_TRAIN_parent_cache',fresh_done=done,total_fresh=1891,
                seconds=time.monotonic()-started)),flush=True)
    if bundle(directory)!=identity:raise ValueError('Read-only retention inference changed teacher bundle')
    targets=make_targets(rows,predictions)
    if len(targets)!=4244 or any(sha(path)!=expected for path,expected in bindings.items()):
        raise ValueError('Complete exact TRAIN target membership and immutable prerequisites required')
    result=dict(status='all4244_augmented_TRAIN_RI_parent_probabilities_frozen_not_fit_admitted',
        qualified_parent_identity=expected,TRAIN_rows=92570,RI_TRAIN_targets=4244,
        reused_original_predictions=2353,newly_inferred_TRAIN_menus=1891,
        source_counts=dict(Counter(r['questions']['action']['src'] for r in selected)),
        complete_input_label_source_menu_membership_checked=True,
        no_VAL_or_test_requests_inferred=True,optimizer_created=False,student_updates=0,new_games=0,
        student_fit_admitted=False,CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.monotonic()-started,allocation=allocation)
    write(OUT/'summary-private.json',result)
    for path in (destination,OUT/'registered-private.json',OUT/'summary-private.json'):bindings[str(path)]=sha(path)
    write(OUT/'completion-binding-private.json',{'files':bindings})
    print(json.dumps({k:v for k,v in result.items() if k!='allocation'}),flush=True)


if __name__=='__main__':main()
