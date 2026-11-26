"""Freeze audited independent basic data with exact original19 replay; no fit."""
from collections import Counter
import copy
import json
from pathlib import Path
import random
import sys
import time
import tomllib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.audit_curriculum_basic_teacher import fingerprint
from tools.curriculum_progression_parent_loss_v1 import MASS
from tools.curriculum_course_order import course_order
from tools.curriculum_ri_coverage_replay_v2 import build,RI
from tools.curriculum_ri_replay_epoch_adapter_v1 import EpochReplay

CONFIG=ROOT/'configs/training/curriculum-ri-augmented-data-v1.json'
PLAN=ROOT/'docs/experiments/plans/curriculum-ri-augmented-data-20261125.md'


def main():
    allocation=resource();started=time.monotonic();config=read(CONFIG)
    out=ROOT/config['out']
    if (out.exists() or config['method']!='audited_basic256_plus_exact_original19_source_union_v1'
            or config['source_mass']!='unchanged_retained19' or config['epochs']!=2
            or config['student_fit_automatic'] is not False or len(config['exposure_seeds'])!=2):
        raise ValueError('Only the separate frozen two-epoch standby union may be prepared once')
    registered=ROOT/config['current_registration'];reg=read(registered)
    args=tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    audit_binding=ROOT/config['supplement_audit'];bound=read(audit_binding)['files']
    native_binding=read(audit_binding.parent/'audit-v1/audit-binding-private.json')['files']
    for path,expected in native_binding.items():
        if path in bound and bound[path]!=expected:raise ValueError('Conflicting audit source identity')
        bound[path]=expected
    if any(sha(path)!=expected for path,expected in bound.items()):
        raise ValueError('All completed native/encoder/split audit evidence must remain unchanged')
    supplement=read(audit_binding.parent/'audit-v1/summary-private.json')
    final_audit=read(audit_binding.parent/'audit-driver-summary-private.json')
    if (final_audit['status']!='independent_basic256_native_encoder_and_full_current_course_exclusions_passed_not_fit_admitted'
            or final_audit['checked_complete_source_games']!=256 or final_audit['student_fit_admitted'] is not False
            or any(any(v.values()) for v in final_audit['full_current_TRAIN_VAL_overlap_counts'].values())):
        raise ValueError('All256 completed sources and complete current-course exclusions required')
    files=dict(bound)
    for path in (CONFIG,PLAN,Path(__file__),registered,audit_binding,
                 ROOT/'tools/curriculum_ri_coverage_replay_v2.py',
                 ROOT/'tests/core/test_curriculum_ri_coverage_replay_v2.py',
                 ROOT/'tools/curriculum_course_order.py',ROOT/'tools/curriculum_ri_replay_epoch_adapter_v1.py'):
        files[str(path)]=sha(path)
    pools={};cohorts={};sources={};out.mkdir(parents=True)
    for split,key in (('train','data'),('validation','eval_data')):
        original=Path(args[key]);spec=supplement['files'][split];extra=ROOT/spec['path']
        if (sha(original)!=reg['inputs'][str(original)] or sha(extra)!=spec['sha256']
                or spec['records']!=config['expected_supplement'][split]):
            raise ValueError('Exact current originals and complete audited supplement required')
        files[str(original)]=sha(original);files[str(extra)]=spec['sha256']
        result=[];original_ids=[];extra_ids=[];destination=out/f'{split}-union-private.jsonl'
        with destination.open('wb') as stream:
            for line in original.open('rb'):
                if not line.endswith(b'\n'):raise ValueError('Preserve exact original JSONL line bytes')
                row=json.loads(line);result.append(row);original_ids.append(row['_meta']['id']);stream.write(line)
            if len(result)!=config['expected_original'][split]:raise ValueError('Full original cohort required')
            for line in extra.open():
                raw=json.loads(line);row=copy.deepcopy(raw);q=row['questions']['action'];meta=row['_meta']
                category=meta['curriculum_stratum']
                if (category not in ('plain','riichi_yes','riichi_no') or q['src']!='curriculum1:'+category
                        or meta['split']!=split):raise ValueError('Unchanged audited teacher source class/split required')
                q['src']='curriculum2:basic:teacher:'+category
                restored=copy.deepcopy(row);restored['questions']['action']['src']=raw['questions']['action']['src']
                if restored!=raw:raise ValueError('Supplement may change only its private loss-source tag')
                result.append(row);extra_ids.append(meta['id'])
                stream.write((json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n').encode())
        if len(result)!=config['expected_union'][split] or len(extra_ids)!=config['expected_supplement'][split]:
            raise ValueError('All original and supplemental decisions must be present')
        ids=[r['_meta']['id'] for r in result]
        if len(set(ids))!=len(ids) or {r['questions']['action']['src'] for r in result}!=set(MASS):
            raise ValueError('Distinct unchanged nineteen-source union required')
        pools[split]=result;cohorts[split]=dict(original=original_ids,supplement=extra_ids)
        sources[split]=dict(Counter(r['questions']['action']['src'] for r in result))
        files[str(destination)]=sha(destination)
    for field in ('ids','groups','inputs'):
        def values(rows):
            return {r['_meta']['id'] if field=='ids' else r['_meta']['group_id'] if field=='groups' else fingerprint(r) for r in rows}
        if values(pools['train'])&values(pools['validation']):raise ValueError('Cross-split union overlap: '+field)
    train=pools['train'];categories=[r['questions']['action']['src'] for r in train]
    if sum(s in RI for s in categories)!=config['expected_RI_TRAIN']:
        raise ValueError('All4244 RI TRAIN opportunities must survive')
    rng=random.Random(config['baseline_seed']);plans=[]
    for number,seed in enumerate(config['exposure_seeds'],1):
        baseline=course_order(categories,rng,511,min_tail_fraction=0.)
        plan=build(categories,baseline,seed,batch=config['accumulation'],
            max_repeat=config['max_repeat'],max_weight=config['max_example_weight'])
        plan['summary']['epoch']=number;plans.append(plan)
        path=out/f'epoch-{number:02d}-private.json';write(path,plan);files[str(path)]=sha(path)
    EpochReplay(train,plans) # Reuse the actual generic epoch adapter's strict constructor.
    cohort_path=out/'cohort-ids-private.json';write(cohort_path,cohorts);files[str(cohort_path)]=sha(cohort_path)
    if any(sha(path)!=expected for path,expected in files.items()):raise ValueError('Frozen inputs changed')
    summary=dict(status='audited_augmented19_union_and_full_coverage_replay_frozen_not_fit_registered',
        unique_rows={s:len(v) for s,v in pools.items()},RI_TRAIN_records=config['expected_RI_TRAIN'],
        source_counts=sources,unchanged_source_masses=MASS,
        original_line_bytes_preserved=True,supplement_only_loss_tag_changed=True,
        original_VAL_and_new_VAL_cohorts_retained_separately=True,
        cross_split_ID_group_visible_input_overlap=0,
        epochs=[p['summary'] for p in plans],total_optimizer_updates=sum(p['summary']['optimizer_updates'] for p in plans),
        total_physical_exposures=sum(len(p['indices']) for p in plans),
        current_trial_changed=False,model_loaded=False,student_updates=0,new_games=0,
        combined_actual_token_admission_and_parent_target_cache_pending=True,
        student_fit_admitted=False,seconds=time.monotonic()-started,allocation=allocation)
    write(out/'summary-private.json',summary);files[str(out/'summary-private.json')]=sha(out/'summary-private.json')
    write(out/'completion-binding-private.json',{'files':files})
    print(json.dumps({k:v for k,v in summary.items() if k not in ('allocation','epochs','source_counts','unchanged_source_masses')}),flush=True)
    print(json.dumps(dict(epoch_summaries=[p['summary'] for p in plans])),flush=True)


if __name__=='__main__':main()
