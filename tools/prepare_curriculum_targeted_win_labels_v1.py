"""Register paired complete-match values for the actual fresh native sources."""
from collections import Counter
import json,os,socket,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read,sha
from tools.curriculum_rank_win_controls_v1 import METHOD

PUBLIC_FILES=('tools/prepare_curriculum_targeted_win_labels_v1.py','tools/collect_curriculum_targeted_win_labels_v1.py',
    'tools/merge_curriculum_targeted_win_labels_v1.py','configs/training/curriculum-targeted-win-labels-v1.json',
    'cluster/curriculum-targeted-win-labels-v1.slurm','docs/experiments/plans/curriculum-targeted-win-labels-20261123.md')

def prepare():
    if (not os.environ.get('SLURM_JOB_ID') or not (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
        or not socket.gethostname().startswith('ws-')):raise RuntimeError('Actual bounded GPU resources required')
    config=ROOT/'configs/training/curriculum-targeted-win-labels-v1.json';c=read(config)
    if (c['method']!=METHOD or c['expected_views']!={'train':72,'validation':36} or c['views_per_shard']!=27
        or c['shards']!=4 or c['worlds_per_block']!=16 or c['student_fit']):
        raise ValueError('Frozen108-view32-world target recipe required')
    ready_path=ROOT/c['source_ready'];ready=read(ready_path)
    if (ready['status']!='fresh_targeted_late_win_special_native_sources_prepared_no_labels'
        or ready['source_quarantine']!=0 or not ready['heldout_RI_ankan_kakan_win_windows_actual']
        or not ready['all_native_sources_reconstructed'] or ready['labels_generated'] or ready['student_updates']):
        raise ValueError('Actual complete fresh108-view native source acquisition required')
    generation=read(ready_path.parent/'registered-plans-private.json')
    source_path=ROOT/generation['config']['source_registration'];source=read(source_path)
    sys.path.insert(0,source['source']);sys.path.insert(0,source['training_config']['upstream_source'])
    from importlib.metadata import version
    import riichienv._riichienv as native
    from jevmahjong.evaluation.unified_contract import bundle
    from tools.curriculum_placement_input_v1 import serving_request
    from tools.audit_curriculum_reused_terminal_v1 import fingerprint
    if version('riichienv')!='0.4.11':raise ValueError('Pinned verified native engine required')
    files=dict(read(ready_path.parent/'completion-binding-private.json')['files'])
    active=read(ready_path.parent/'completion-binding-private.json')['active_fit_files']
    selected=[];fps={p:set() for p in ('train','validation')};groups={p:set() for p in fps}
    for split in fps:
        path=ROOT/ready['files'][split]['path']
        if sha(path)!=ready['files'][split]['sha256']:raise ValueError('Fresh native source changed')
        files[str(path.resolve())]=sha(path)
        for line in path.open():
            row=json.loads(line);m=row['_meta'];criteria=row['questions']['action']['criteria']
            if m['split']!=split or any(k in row['questions']['action'] for k in ('src','label','target')):
                raise ValueError('Fresh source must keep original splits and remain unlabelled')
            if not 2<=len(criteria)<=16:raise ValueError('Complete native menu outside the registered cap')
            public=serving_request(row);fps[split].update((fingerprint(public),fingerprint(public,True)))
            groups[split].add(m['group_id'])
            selected.append(dict(id=m['id'],split=split,group=m['group_id'],physical_id=m['physical_id'],
                category='win',context=m['score_context'],public_rank=m['placement']['rank'],
                bucket=m['targeted_operation']+':'+m['score_context'],legal_menu_size=len(criteria)))
    if dict(Counter(s['split'] for s in selected))!=c['expected_views'] or groups['train'] & groups['validation']:
        raise ValueError('Exact72/36 source views and original whole families required')
    old_path=ROOT/'runs/curriculum-reused-terminal-v1/collection-v1/summary-private.json';old=read(old_path)
    files[str(old_path.resolve())]=sha(old_path)
    for split in fps:
        path=ROOT/old['files'][split]['path']
        if sha(path)!=old['files'][split]['sha256']:raise ValueError('Original reused source changed')
        files[str(path.resolve())]=sha(path);opposite='validation' if split=='train' else 'train'
        for line in path.open():
            row=json.loads(line);public=serving_request(row)
            if row['_meta']['group_id'] in groups[opposite] or {fingerprint(public),fingerprint(public,True)} & fps[opposite]:
                raise ValueError('Fresh source overlaps opposite earlier terminal source/group')
    selected.sort(key=lambda s:(s['split']!='train',s['bucket'],s['id']))
    proof_path=ROOT/'runs/curriculum-full-win-route-v1/verification-v1/summary-private.json'
    if read(proof_path)['status']!='actual_complete_current_win_menu_native_encoder_and_single_forward_verified_not_fit':
        raise ValueError('Actual public-event full-menu route proof required')
    proof_binding=proof_path.parent/'completion-binding-private.json'
    for p,h in read(proof_binding)['files'].items():
        if p in files and files[p]!=h:raise ValueError('Conflicting immutable input')
        files[p]=h
    from tools.prepare_curriculum_rank_win_labels_v1 import HELPERS
    paths=[config,source_path,ready_path,ready_path.parent/'completion-binding-private.json',
        proof_path,proof_binding,Path(native.__file__)]
    paths += [ROOT/n for n in PUBLIC_FILES]+[ROOT/'tools'/n for n in HELPERS]
    paths += [ROOT/'tools/curriculum_rank_win_controls_v1.py',ROOT/'tools/curriculum_rank_nonwin_teacher_v2.py',
        ROOT/'tools/curriculum_targeted_win_source_v1.py',ROOT/'tools/curriculum_rank_win_teacher_v1.py']
    model=bundle(Path(source['curriculum']['reference_checkpoint']))
    if model['identity_sha256']!=source['curriculum']['reference_identity']:raise ValueError('Frozen teacher continuation changed')
    paths += [Path(model['directory'])/n for n in model['files']]
    for path in paths:files[str(path.resolve())]=sha(path)
    for n in PUBLIC_FILES:
        if n.endswith('.py'):compile((ROOT/n).read_text(),str(ROOT/n),'exec')
    if any(sha(p)!=h for p,h in {**files,**active}.items()):raise ValueError('Source or active fit input changed')
    arms=32*sum(s['legal_menu_size'] for s in selected)
    if arms>c['maximum_outer_terminal_arms']:raise ValueError('Prospective current-menu arm budget exceeded')
    out=ROOT/c['out'];out.mkdir(parents=True,exist_ok=False)
    result=dict(status='fresh_targeted_win_operation_views_frozen_before_values',config=c,selected=selected,
        source_registration=str(source_path.resolve()),source_files=ready['files'],files=files,active_fit_files=active,
        continuation_model=model,counts=c['expected_views'],families={p:len(v) for p,v in groups.items()},
        public_menu_arm_budget=arms,source_quarantine=0,labels_generated=0,student_updates=0,
        no_future_value_based_source_selection=True,exact_opposite_prior_terminal_input_group_overlap=0,
        constructed_conditionals_not_natural_match_distribution=True,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'])
    write(out/'registered-private.json',result)
    public={k:v for k,v in result.items() if k not in ('selected','files','active_fit_files','continuation_model','source_files','source_registration','config')}
    write(out/'prepare-summary-public.json',public);print(json.dumps(public),flush=True)

if __name__=='__main__':prepare()

