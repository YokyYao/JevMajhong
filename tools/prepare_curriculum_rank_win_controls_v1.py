"""Freeze broad win controls from unused public score cohorts before values."""
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_rank_win_controls_v1 import METHOD,bucket,select

PUBLIC_FILES=('tools/curriculum_rank_win_controls_v1.py','tools/prepare_curriculum_rank_win_controls_v1.py',
    'tools/collect_curriculum_rank_win_controls_v1.py','tools/merge_curriculum_rank_win_controls_v1.py',
    'tests/core/test_curriculum_rank_win_controls_v1.py','configs/training/curriculum-rank-win-controls-v1.json',
    'cluster/curriculum-rank-win-controls-v1.slurm','docs/experiments/plans/curriculum-rank-win-controls-20261123.md')


def prepare():
    if (not os.environ.get('SLURM_JOB_ID') or not (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
        or not socket.gethostname().startswith('ws-')):raise RuntimeError('Bounded scheduler GPU resources required')
    config=ROOT/'configs/training/curriculum-rank-win-controls-v1.json';c=read(config)
    if (c['method']!=METHOD or c['expected_selected']!={'train':96,'validation':22}
        or c['maximum_current_actions']!=16 or c['views_per_shard']!=30 or c['shards']!=4
        or c['worlds_per_block']!=16 or c['bootstrap_samples']!=512 or c['student_fit']):
        raise ValueError('Predeclared broad control cohort and fixed32-world recipe required')
    root=ROOT/c['data_root'];audit_path=root/'input-audit-v1/summary-private.json';audit=read(audit_path)
    if audit['status']!='reused_terminal_native_encoder_split_and_public_score_audit_passed':raise ValueError('Completed input audit required')
    reg_path=ROOT/c['source_registration'];reg=read(reg_path)
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.base_lora import package_digest
    from tools.audit_curriculum_reused_terminal_v1 import restore
    from tools.audit_curriculum_endgame_cases import public_solo_win
    if package_digest(Path(reg['source'])/'jevmahjong')!=reg['source_sha256']:raise ValueError('Pinned runtime changed')
    ready_path=root/'collection-v1/summary-private.json';ready=read(ready_path);rows=[];paths=[config,reg_path,audit_path,ready_path]
    detail_path=root/'input-audit-v1/public-score-details-private.jsonl';paths.append(detail_path)
    details={d['id']:d for d in map(json.loads,detail_path.open())}
    old_path=ROOT/'runs/curriculum-rank-win-labels-v1/registered-private.json';old=read(old_path);paths.append(old_path)
    excluded={s['id'] for s in old['selected']};scores={rid:d['public_solo_score'] for rid,d in details.items()}
    for split in ('train','validation'):
        path=ROOT/ready['files'][split]['path'];paths.append(path)
        if sha(path)!=ready['files'][split]['sha256']:raise ValueError('Audited source changed')
        for line in path.open():
            row=json.loads(line)
            if row['_meta']['category']!='win':continue
            m=row['_meta'];detail=details[m['id']]
            if detail['split']!=split or detail['group_id']!=m['group_id']:raise ValueError('Score/source lineage differs')
            rows.append(row)
    chosen=select(rows,scores,excluded,c['quotas']);selected=[];special=Counter()
    for name,row in chosen:
        m=row['_meta'];env,obs=restore(row);actual=public_solo_win(env,obs)
        if actual!=scores[m['id']] or bucket(row,actual)!=name:raise ValueError('Public score proxy failed actual native reconstruction')
        keys=row['questions']['action']['criteria'];kinds={v['type'] for v in keys.values()}
        if not 2<=len(keys)<=16 or sum(v['type']=='hora' for v in keys.values())!=1:
            raise ValueError('Actual full win menu exceeds the registered budget')
        for op in kinds & {'reach','ankan','kakan','daiminkan'}:special[m['split']+':'+op]+=1
        selected.append(dict(id=m['id'],split=m['split'],group=m['group_id'],physical_id=m['physical_id'],
            category='win',context=m['score_context'],public_rank=m['placement']['rank'],
            bucket=name,legal_menu_size=len(keys),public_score_proxy=actual))
    selected.sort(key=lambda s:(s['split']!='train',s['bucket'],s['id']))
    counts=dict(Counter(s['split'] for s in selected))
    groups={p:{s['group'] for s in selected if s['split']==p} for p in ('train','validation')}
    if counts!=c['expected_selected'] or groups['train'] & groups['validation'] or len({s['id'] for s in selected})!=118:
        raise ValueError('Original whole-family split and exact118 unique views required')
    replay_path=ROOT/'runs/curriculum-yaku-scenarios-v1/data-v2/data-ready-private.json';replay=read(replay_path);paths.append(replay_path)
    for split in ('train','validation'):
        path=ROOT/replay['files'][split]['path'];paths.append(path)
        if sha(path)!=replay['files'][split]['sha256']:raise ValueError('Current replay changed')
        other=groups['validation' if split=='train' else 'train']
        if any(json.loads(line)['_meta']['group_id'] in other for line in path.open()):
            raise ValueError('Opposite current replay source-family overlap')
    model=bundle(Path(reg['curriculum']['reference_checkpoint']))
    if model['identity_sha256']!=reg['curriculum']['reference_identity']:raise ValueError('Frozen v8 changed')
    paths += [Path(model['directory'])/n for n in model['files']]
    paths += [ROOT/n for n in PUBLIC_FILES]
    from tools.prepare_curriculum_rank_win_labels_v1 import HELPERS
    paths += [ROOT/'tools'/n for n in HELPERS]
    paths += [ROOT/'tools/curriculum_rank_win_teacher_v1.py',ROOT/'tools/curriculum_rank_nonwin_teacher_v2.py',
        ROOT/'tools/curriculum_terminal_rank_priority_v1.py',ROOT/'tools/audit_curriculum_endgame_cases.py']
    for name in PUBLIC_FILES:
        if name.endswith('.py'):compile((ROOT/name).read_text(),str(ROOT/name),'exec')
    active_path=ROOT/'runs/curriculum-progression-v4/launch-binding-private.json';paths.append(active_path)
    active=read(active_path)['files'];files={str(p.resolve()):sha(p) for p in paths}
    if any(sha(p)!=h for p,h in {**files,**active}.items()):raise ValueError('Frozen inputs changed')
    arms=32*sum(s['legal_menu_size'] for s in selected)
    if arms>c['maximum_outer_terminal_arms']:raise ValueError('Public preregistered arm budget exceeded')
    out=ROOT/c['out'];out.mkdir(parents=True,exist_ok=False)
    result=dict(status='public_rank_win_controls118_views_frozen_before_values',config=c,
        selected=selected,source_registration=str(reg_path.resolve()),source_files=ready['files'],
        files=files,active_fit_files=active,continuation_model=model,counts=counts,
        families={p:len(v) for p,v in groups.items()},public_menu_arm_budget=arms,
        special_continuation_views=dict(special),all_selected_scores_reconstructed=True,
        no_overlap_with_existing263_win_view_ids=True,source_families_may_repeat_in_matched_score_contexts=True,
        selection_uses_public_no_ura_solo_proxy_not_terminal_outcomes=True,
        labels_generated=0,student_updates=0,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'])
    write(out/'registered-private.json',result)
    public={k:v for k,v in result.items() if k not in ('selected','files','active_fit_files','continuation_model','source_files','source_registration','config')}
    write(out/'prepare-summary-public.json',public);print(json.dumps(public),flush=True)


if __name__=='__main__':prepare()

