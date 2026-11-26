"""Reconcile a cached RI-validation prefix and same-case decision transitions.

No model inference, games, parameter updates or threshold tuning. Raw case
transitions remain private; aggregate counts are descriptive development data.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_riichi_control import require_registered,profile
from tools.curriculum_riichi_mechanisms import score_rows
from tools.curriculum_basic_repair import grade
from tools.plot_curriculum_fixed_grid import complete_rows


def transitions(before,after):
    def index(cases):
        result={r['id']:r for r in cases}
        if len(result)!=len(cases) or not result:raise ValueError('Distinct nonempty case IDs required')
        return result
    a,b=index(before),index(after)
    if set(a)!=set(b):raise ValueError('Same validation cases required')
    counts={s:Counter() for s in ('legacy','teacher')};private=[]
    for identifier,left in a.items():
        right=b[identifier]
        if any(left[k]!=right[k] for k in ('group','source','truth','discard_choices')):
            raise ValueError('Case identity, label, source or legal menu changed')
        if left['source'] not in counts:raise ValueError('Unexpected source')
        if any(type(r[k]) is not bool for r in (left,right) for k in ('truth','raw_argmax')):
            raise ValueError('Boolean truth and action required')
        x,y=left['raw_argmax']==left['truth'],right['raw_argmax']==right['truth']
        status=('correct' if x else 'wrong')+'_to_'+('correct' if y else 'wrong')
        counts[left['source']][status]+=1
        private.append(dict(id=identifier,source=left['source'],truth=left['truth'],
            before_reach=left['raw_argmax'],after_reach=right['raw_argmax'],transition=status))
    result={}
    for source in (*counts,'pooled'):
        c=counts[source] if source in counts else sum(counts.values(),Counter())
        keys=('correct_to_correct','correct_to_wrong','wrong_to_correct','wrong_to_wrong')
        n=sum(c[k] for k in keys)
        if not n:raise ValueError('Both sources required')
        result[source]={k:c[k] for k in keys}|dict(windows=n,
            changed=c['correct_to_wrong']+c['wrong_to_correct'],
            net_correct_change=c['wrong_to_correct']-c['correct_to_wrong'],
            correct_before=c['correct_to_correct']+c['correct_to_wrong'],
            correct_after=c['correct_to_correct']+c['wrong_to_correct'])
    return result,private


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def audit(registered,step,out):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-','login'))):raise RuntimeError('Scheduled GPU required')
    if step not in range(76,533,76):raise ValueError('Actual interim76-step validation points only; final608 has its own full audit')
    reg=read(registered);oldreg=require_registered(reg)
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import target_cross_entropy
    from jevmahjong.evaluation.unified_contract import bundle
    out.mkdir(parents=True,exist_ok=False);files=dict(reg['inputs'])
    for p in (registered,Path(__file__),ROOT/'tools/curriculum_riichi_mechanisms.py',ROOT/'tools/export_curriculum_riichi_curves_v1.py'):
        files[str(p.resolve())]=sha(p)
    assert all(sha(p)==h for p,h in files.items())
    data=read(ROOT/reg['curriculum']['data_root']/'data-ready.json')
    path=ROOT/data['stages']['stage1']['validation']['path'];files[str(path)]=sha(path)
    records=[json.loads(line) for line in path.open()];points={};mechanisms={};changes={};identities={}
    new,old=(ROOT/r['curriculum']['out'] for r in (reg,oldreg))
    for name,folder in [('hard_ce',old),('max_class',new)]:
        points[name]={};mechanisms[name]={};changes[name]={};identities[name]={};previous=None
        for actual_step in range(0,step+1,76):
            prediction=folder/'training-run/training_eval'/f'step-{actual_step:06d}'/'development/rows.json'
            files[str(prediction)]=sha(prediction)
            point=next(p for p in read(folder/'validation-curve.json') if p['step']==actual_step)
            preds=read(prediction)
            p=profile(records,preds,reg['stage1_loss_mass'],dict(weighted_target_ce=point['natural_target_ce']),target_cross_entropy)
            for key in ('stratum_ce','stratum_accuracy','source_stratum_ce','source_stratum_accuracy','riichi_by_source','riichi_type_counts'):
                assert p[key]==point[key],key
            assert abs(p['weighted_target_ce']-point['weighted_target_ce'])<1e-12
            if name=='max_class':assert abs(p['weighted_class_objective']-point['weighted_class_objective'])<1e-12
            scores,cases=score_rows(records,preds)
            assert len(cases)==632
            for source,c in p['riichi_by_source'].items():
                assert scores['sources'][source]['confusion']['raw_argmax']=={k:c.get(k,0) for k in ('tp','fn','tn','fp')}
            mechanisms[name][str(actual_step)]={s:dict(
                margin_auc=v['argmax_logit_margin_auc'],class_probability_auc=v['class_probability_auc'],
                mean_p_reach_accept=v['cohorts']['teacher_yes']['p_riichi']['mean'],
                mean_p_reach_decline=v['cohorts']['teacher_no']['p_riichi']['mean']) for s,v in scores['sources'].items()}
            if previous is not None:
                key=f'{actual_step-76}-{actual_step}';aggregate,detail=transitions(previous,cases)
                for source,c in aggregate.items():
                    left=previous_point['riichi_type_counts'] if source=='pooled' else previous_point['riichi_by_source'][source]
                    right=p['riichi_type_counts'] if source=='pooled' else p['riichi_by_source'][source]
                    assert c['correct_before']==left.get('tp',0)+left.get('tn',0)
                    assert c['correct_after']==right.get('tp',0)+right.get('tn',0)
                changes[name][key]=aggregate;write(out/f'{name}-{key}-case-transitions-private.json',detail)
            previous=cases;previous_point=p
            keys=('weighted_target_ce','weighted_class_objective','stratum_ce','stratum_accuracy','riichi_by_source','riichi_type_counts')
            points[name][str(actual_step)]={k:p[k] for k in keys}
            identities[name][str(actual_step)]=bundle(folder/'training-run/checkpoints'/f'step-{actual_step:06d}')['identity_sha256']
        if name=='max_class':current_quality=grade(p)
    # Prefix source exposure and objectives are reconciled against the fixed old
    # control, including a partial final epoch. Never pretend generated orders
    # have all been consumed.
    previous={r['local_step']:r for r in complete_rows(old/'metrics/train_metrics.jsonl') if r['local_step']<=step}
    updates=[r for r in complete_rows(new/'metrics/train_metrics.jsonl') if r['local_step']<=step]
    assert [r['local_step'] for r in updates]==list(range(1,step+1));expected=[]
    for epoch in range(1,(step+151)//152+1):
        a=new/'epoch-order-private'/f'epoch-{epoch:02d}.json';b=old/'epoch-order-private'/f'epoch-{epoch:02d}.json'
        receipt=read(a);assert receipt==read(b) and receipt['each_record_exactly_once'] and receipt['copies_added']==0
        expected.extend(receipt['source_counts_by_update'])
        for path in (a,b):files[str(path)]=sha(path)
    counts=Counter();loss_error=0.;ce_error=0.;intervals={}
    for row,wanted in zip(updates,expected):
        assert row['source_counts']==previous[row['local_step']]['source_counts']==wanted
        n=sum(wanted.values());counts.update(wanted)
        weighted={s:n*reg['stage1_loss_weights'][s] for s,n in wanted.items()}
        loss=sum(weighted[s]*row['source_objective'][s] for s in wanted)/n
        ce=sum(weighted[s]*row['source_ce'][s] for s in wanted)/n
        loss_error=max(loss_error,abs(loss-row['loss']));ce_error=max(ce_error,abs(ce-row['ce']))
        interval=(row['local_step']-1)//76;totals=intervals.setdefault(interval,[0.,0.])
        totals[0]+=sum(v for s,v in weighted.items() if s.endswith(':riichi_yes'))
        totals[1]+=sum(v for s,v in weighted.items() if s.endswith(':riichi_no'))
    assert sum(counts.values())==updates[-1]['records_seen'] and loss_error<5e-6 and ce_error<5e-6
    # Anchor the already independently audited304 result rather than silently
    # replacing its interpretation with a new implementation.
    if step>=304:
        anchor=ROOT/'runs/curriculum-riichi-matched304-v1/audit-v1/summary-private.json'
        known=read(anchor);files[str(anchor)]=sha(anchor)
        for name in points:
            assert points[name]['304']==known['validation'][name]
            assert mechanisms[name]['304']==known['source_mechanisms'][name]
            assert identities[name]['304']==known['checkpoint_identities'][name]
    from tools.export_curriculum_riichi_curves_v1 import export
    export(registered,out/'curves',step)
    assert all(sha(p)==h for p,h in files.items())
    result=dict(status='cached_validation_prefix_and_same_case_RI_transitions_reconciled',step=step,
        validation=points,source_mechanisms=mechanisms,same_case_transitions=changes,
        checkpoint_identities=identities,actual_exposures=sum(counts.values()),actual_source_exposure=dict(counts),
        actual_updates_and_generated_orders_match_original=True,complete_epochs=step//152,
        partial_epoch_updates=step%152,maximum_objective_reconstruction_error=loss_error,
        maximum_original_ce_reconstruction_error=ce_error,
        actual_RI_positive_loss_mass_by76_updates={f'{k*76+1}-{(k+1)*76}':v[0]/sum(v) for k,v in intervals.items()},
        current_source_quality=current_quality,
        no_new_model_inference_games_updates_threshold_tuning_or_parent_admission=True,
        no_intermediate_candidate_selection=True,raw_details_private=True)
    write(out/'summary-private.json',result)
    write(out/'binding-private.json',dict(files=files,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(status=result['status'],step=step,actual_exposures=result['actual_exposures'],latest=points['max_class'][str(step)],latest_transitions=changes['max_class'][f'{step-76}-{step}'])),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--registered',type=Path,required=True)
    parser.add_argument('--step',type=int,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();audit(args.registered.resolve(),args.step,args.out.resolve())
