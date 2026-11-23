"""Final-only cached RI decoder contrast; no threshold search or policy change.

Reuse the previously defined raw-argmax versus fixed-full-class0.5 diagnostic.
Neither hypothetical label agreement nor its ranking scores are native wins.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_riichi_control import require_registered
from tools.curriculum_riichi_mechanisms import score_rows


def require_final(completion,evidence):
    if (completion.get('status')!='stage1_fit_and_curves_complete'
        or completion.get('optimizer_steps')!=608 or completion.get('actual_records_seen')!=57140
        or completion.get('curve_steps')!=[0,152,304,456,608]
        or evidence.get('status')!='stage1_final_evidence_audit_passed'
        or evidence.get('optimizer_steps')!=608 or evidence.get('real_records_seen')!=57140
        or evidence.get('source_and_data_hashes_unchanged') is not True
        or evidence.get('final_summary',{}).get('matches')!=64
        or evidence.get('final_summary',{}).get('seed_groups')!=8):
        raise ValueError('Completed608/full64 and independent final integrity evidence required')
    # A failed source grade is allowed for diagnosis, never admitted as a parent.


def contrast(records,predictions):
    measured,_=score_rows(records,predictions)
    keep=('windows','teacher_acceptance','class_probability_auc','argmax_logit_margin_auc',
        'mean_class_nll','mean_conditional_discard_nll','mean_normalized_action_nll',
        'max_nll_decomposition_error','confusion','fixed_decoder_disagreements')
    return dict(sources={s:{k:v[k] for k in keep} for s,v in measured['sources'].items()},
        pooled={k:measured['pooled'][k] for k in keep},
        fixed_threshold=.5,threshold_search=False,actual_policy_decisions_changed=False,
        diagnostic_label_agreement_not_native_strength=True)


def audit(registered,out):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-','login'))):raise RuntimeError('Scheduled GPU required')
    reg=read(registered);old=require_registered(reg)
    if any(sha(p)!=h for p,h in reg['inputs'].items()):raise ValueError('Frozen registered inputs changed')
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle
    data=read(ROOT/reg['curriculum']['data_root']/'data-ready.json');spec=data['stages']['stage1']['validation']
    path=ROOT/spec['path']
    if sha(path)!=spec['sha256']:raise ValueError('Actual validation changed')
    records=[json.loads(line) for line in path.open()]
    if len(records)!=3266:raise ValueError('All3266 actual validation menus required')
    out.mkdir(parents=True,exist_ok=False);files=dict(reg['inputs']);results={};identities={}
    for p in (registered,path,Path(__file__),ROOT/'tools/curriculum_riichi_mechanisms.py',
        ROOT/'docs/experiments/plans/curriculum-riichi-final-decoder-contrast-20261123.md'):
        files[str(p.resolve())]=sha(p)
    for name,current in [('hard_ce',old),('max_class',reg)]:
        fit=ROOT/current['curriculum']['out'];evidence_path=fit.parent/'final-evidence-private.json'
        completion=fit/'completion.json';evidence=read(evidence_path);require_final(read(completion),evidence)
        predictions=fit/'training-run/training_eval/step-000608/development/rows.json'
        curve=fit/'validation-curve.json';point=next(p for p in read(curve) if p['step']==608)
        for p in (evidence_path,completion,predictions,curve):files[str(p)]=sha(p)
        result=contrast(records,read(predictions));assert result['pooled']['windows']==632
        for source,expected in point['riichi_by_source'].items():
            assert result['sources'][source]['confusion']['raw_argmax']=={k:expected.get(k,0) for k in ('tp','fn','tn','fp')}
        identity=bundle(fit/'training-run/checkpoints/step-000608')['identity_sha256']
        reloads=[r for r in evidence['saved_reloads'] if r['step']==608]
        assert len(reloads)==1 and reloads[0]['menus']==3266 and reloads[0]['identity']==identity
        results[name]=result;identities[name]=identity
    if any(sha(p)!=h for p,h in files.items()):raise ValueError('Diagnostic inputs changed')
    def write(p,v):
        with p.open('x') as stream:json.dump(v,stream,indent=2);stream.write('\n')
    result=dict(status='two_completed608_cached_fixed_decoder_contrasts_not_policy_evaluation',
        step=608,models=results,checkpoint_identities=identities,complete_menus_per_model=3266,
        RI_windows_per_model=632,source_quality_not_overridden=True,parent_admitted=False,
        no_new_inference_games_labels_optimizer_updates_or_threshold_selection=True,
        hypothetical_decoder_not_installed=True,raw_cases_and_predictions_private=True,
        final_reserved_grid_unused=True,goal_achieved=False)
    write(out/'summary-private.json',result)
    write(out/'binding-private.json',dict(files=files,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.registered.resolve(),a.out.resolve())
