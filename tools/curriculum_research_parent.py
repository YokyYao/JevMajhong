"""Explicit research-only amendment; never a replacement for old audit grades."""
import copy
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from tools.curriculum_parent import ROOT,ParentNotReady,read,sha,require_final_integrity

METHOD='curriculum_registered_final_research_v2'
POLICY=ROOT/'configs/training/curriculum-research-parent-v2.json'
POLICY_SHA='6ceff847b9ae2362ba079fe42372034c9f7d95f3f2b73b7921b061aa75b3c288'
REGISTRAR_SHA='1fcaca663c42a5d6ac098684ccaaa6d48b236b753a1294ca310be66b0a6d4ee3'


def checked_policy(policy):
    if sha(POLICY)!=POLICY_SHA or policy!=read(POLICY):
        raise ParentNotReady('The before-final research policy changed')
    return policy


def load_bound_policy(binding_file,registered):
    binding_file=Path(binding_file).resolve();binding=read(binding_file)
    if (binding.get('status')!='research_parent_amendment_bound_before_final_games'
        or binding.get('development_inspection_amendment_not_original_preregistration') is not True
        or binding.get('original_stage1_gate_and_audits_unchanged') is not True):
        raise ParentNotReady('Explicit prospective research-policy binding required')
    files=dict(binding['bound_files'])
    for path,h in files.items():
        if sha(path)!=h:raise ParentNotReady('Research-policy evidence changed: '+path)
    if files.get(str(POLICY.resolve()))!=POLICY_SHA or files.get(str(Path(registered).resolve()))!=sha(registered):
        raise ParentNotReady('Policy or candidate registration is not in the before-final binding')
    policy=checked_policy(read(POLICY));reg=read(registered)
    registrar=ROOT/'tools/register_curriculum_research_parent.py'
    if files.get(str(registrar.resolve()))!=REGISTRAR_SHA:
        raise ParentNotReady('The original before-final registrar is not bound')
    reference=[Path(p) for p in files if p.endswith('/reference-riichi-gate-private.json')]
    baseline=[Path(p) for p in files if p.endswith('/step-001162/summary.json')]
    registrations=[Path(p) for p in files if p.endswith('/fit-plan/registered.json')]
    if len(reference)!=1 or len(baseline)!=1 or len(registrations)!=2:
        raise ParentNotReady('Reference diagnostics or both candidate registrations are missing')
    ref=read(reference[0]);old=read(baseline[0]);limits=policy['complete_match_development']
    if (ref.get('status')!='reference_riichi_gate_diagnostic_complete' or ref.get('windows')!=107
        or ref.get('counts')!=policy['riichi']['reference_counts']
        or ref.get('checkpoint_sha256')!=policy['riichi']['reference_identity']
        or old.get('matches')!=64 or old.get('seed_groups')!=8
        or old.get('checkpoint_sha256')!=limits['baseline_identity']
        or old.get('top2_rate')!=limits['min_top2_rate'] or old.get('net_points')!=limits['min_net_points']
        or old.get('dealin_hand_rate')!=limits['max_dealin_hand_rate']
        or {read(p)['curriculum']['run_id'] for p in registrations}!=set(policy['applies_to_run_ids'])):
        raise ParentNotReady('Frozen amendment diagnostics differ from the declared controls')
    if (len(binding['observations'])!=2 or {r['run_id'] for r in binding['observations']}!=set(policy['applies_to_run_ids'])
        or any(r['registered_final']!=584 or not 0<=r['observed_updates']<584
               or r.get('final584_games_not_started') is not True for r in binding['observations'])):
        raise ParentNotReady('Both research candidates need a before-final observation')
    rows=[r for r in binding['observations'] if r['run_id']==reg['curriculum']['run_id']]
    if (len(rows)!=1 or rows[0]['registered_final']!=reg['registered_final_step']
        or not 0<=rows[0]['observed_updates']<rows[0]['registered_final']
        or rows[0].get('final584_games_not_started') is not True):
        raise ParentNotReady('Research amendment was not bound before this candidate final')
    files[str(binding_file)]=sha(binding_file);files[str(Path(__file__).resolve())]=sha(Path(__file__))
    return policy,files


def require_research_candidate(reg,completion,audit,validation,identity,policy):
    checked_policy(policy)
    accept,decline=require_final_integrity(reg,completion,audit,validation,identity)
    c=reg['curriculum'];games=policy['complete_match_development'];basic=policy['basic_retention']
    if (c['run_id'] not in policy['applies_to_run_ids'] or c['candidate_rule']!=policy['candidate_rule']
        or reg['registered_final_step']!=584 or c['curve_seed_start']!=games['seed_start']
        or c['opponent_profiles']!=games['profiles'] or c['stage_end_seed_groups']!=games['seed_groups']):
        raise ParentNotReady('Candidate differs from the prospectively declared research grid/budget')
    rc=validation['riichi_type_counts'];reference=policy['riichi']['reference_counts']
    if (rc.get('tp',0)+rc.get('fn',0)!=reference['tp']+reference['fn']
        or rc.get('tn',0)+rc.get('fp',0)!=reference['tn']+reference['fp']):
        raise ParentNotReady('Research riichi validation population changed')
    baseline_balanced=.5*(reference['tp']/(reference['tp']+reference['fn'])+reference['tn']/(reference['tn']+reference['fp']))
    balanced=.5*(accept+decline)
    reaches=rc.get('tp',0)+rc.get('fp',0);discards=rc.get('tn',0)+rc.get('fn',0)
    if (not reaches or not discards or balanced<baseline_balanced
        or validation['stratum_accuracy']['plain']<basic['min_plain_exact']
        or validation['stratum_ce']['plain']>basic['plain_ce_reference']*basic['max_plain_ce_factor']):
        raise ParentNotReady('Research parent loses basic retention, riichi balance or both-type coverage')
    summary=audit.get('final_summary',{})
    if summary.get('matches')!=games['matches'] or summary.get('seed_groups')!=games['seed_groups']:
        raise ParentNotReady('Complete final64 development games required; a16-game peak is insufficient')
    for metric,limit,lower in (('top2_rate','min_top2_rate',True),('net_points','min_net_points',True),('dealin_hand_rate','max_dealin_hand_rate',False)):
        actual=summary.get(metric)
        if type(actual) not in (int,float) or not math.isfinite(actual) or (actual<games[limit] if lower else actual>games[limit]):
            raise ParentNotReady('Research parent does not retain full64 development '+metric)
    comparison=audit.get('final_comparison_to_v8',{})
    for field in ('top2','net_points','hand_win','dealin','fourth'):
        row=comparison.get(field,{});interval=row.get('ci95',[]);gain=row.get('gain')
        if (row.get('seed_groups')!=games['seed_groups'] or len(interval)!=2
            or type(gain) not in (int,float) or not math.isfinite(gain)
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in interval)
            or not interval[0]<=gain<=interval[1]):
            raise ParentNotReady('The actual whole-seed paired v8 comparison is incomplete')
    return dict(status='research_continuation_development_criteria_passed',passed=True,method=METHOD,
        policy_sha256=POLICY_SHA,original_stage1_advancement=copy.deepcopy(audit['stage1_advancement']),
        riichi_acceptance=accept,riichi_correct_decline=decline,riichi_balanced_accuracy=balanced,
        reference_riichi_balanced_accuracy=baseline_balanced,final_development_summary=copy.deepcopy(summary),
        admission_is_not_stage1_mastery_or_v8_strength_proof=True,automatic_release=False)


if __name__=='__main__':
    import argparse,json,os,socket
    p=argparse.ArgumentParser(description='Verify the actual before-final policy binding without admitting a parent')
    p.add_argument('--binding',type=Path,required=True);p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-','login'))
    policy,files=load_bound_policy(a.binding,a.registered)
    result=dict(status='actual_research_parent_policy_binding_verified',node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'],policy_sha256=POLICY_SHA,bound_files=len(files),
        registration=str(a.registered.resolve()),parent_admitted=False,child_fit_started=False)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
