"""Reconcile complete-epoch RI snapshots and cached matched16 development games.

Only152/304/456 are supported. Final608 has a separate independent full audit.
No model inference, policy games, fitting or threshold selection is performed.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_riichi_control import require_registered,profile
from tools.curriculum_basic_repair import grade
from tools.curriculum_riichi_mechanisms import score_rows
from tools.plot_curriculum_fixed_grid import complete_rows


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def audit(registered,step,out,export_curves=True):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-','login'))):raise RuntimeError('Scheduled GPU required')
    if step not in (152,304,456):raise ValueError('Only the actual complete-epoch matched16 points supported')
    reg=read(registered);control=require_registered(reg)
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import target_cross_entropy
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevmahjong.evaluation.unified_contract import bundle
    out.mkdir(parents=True,exist_ok=False);files=dict(reg['inputs'])
    files[str(registered.resolve())]=sha(registered);files[str(Path(__file__).resolve())]=sha(__file__)
    assert all(sha(p)==h for p,h in files.items())
    data=read(ROOT/reg['curriculum']['data_root']/'data-ready.json')
    validation_path=ROOT/data['stages']['stage1']['validation']['path']
    records=[json.loads(line) for line in validation_path.open()];files[str(validation_path)]=sha(validation_path)
    new=ROOT/reg['curriculum']['out'];old=ROOT/control['curriculum']['out'];points={};mechanisms={}
    for name,fit in [('hard_ce',old),('max_class',new)]:
        point=next(p for p in read(fit/'validation-curve.json') if p['step']==step)
        pred_path=fit/'training-run/training_eval'/f'step-{step:06d}'/'development/rows.json';files[str(pred_path)]=sha(pred_path)
        predictions=read(pred_path)
        actual=profile(records,predictions,reg['stage1_loss_mass'],dict(weighted_target_ce=point['natural_target_ce']),target_cross_entropy)
        for key in ('stratum_ce','stratum_accuracy','source_stratum_ce','source_stratum_accuracy','riichi_by_source','riichi_type_counts'):
            assert actual[key]==point[key],key
        assert abs(actual['weighted_target_ce']-point['weighted_target_ce'])<1e-12
        if name=='max_class':assert abs(actual['weighted_class_objective']-point['weighted_class_objective'])<1e-12
        snapshot=out/f'{name}-point-private.json';write(snapshot,point);files[str(snapshot)]=sha(snapshot);points[name]=actual
        scored,_=score_rows(records,predictions)
        mechanisms[name]={source:dict(margin_auc=p['argmax_logit_margin_auc'],class_probability_auc=p['class_probability_auc'],
            mean_p_reach_accept=p['cohorts']['teacher_yes']['p_riichi']['mean'],
            mean_p_reach_decline=p['cohorts']['teacher_no']['p_riichi']['mean']) for source,p in scored['sources'].items()}
    previous={r['local_step']:r for r in complete_rows(old/'metrics/train_metrics.jsonl') if r['local_step']<=step}
    updates=[r for r in complete_rows(new/'metrics/train_metrics.jsonl') if r['local_step']<=step]
    assert [r['local_step'] for r in updates]==list(range(1,step+1));expected=[];epochs=step//152
    for epoch in range(1,epochs+1):
        a=new/'epoch-order-private'/f'epoch-{epoch:02d}.json';b=old/'epoch-order-private'/f'epoch-{epoch:02d}.json'
        receipt=read(a);assert receipt==read(b) and receipt['each_record_exactly_once'] and receipt['copies_added']==0
        expected.extend(receipt['source_counts_by_update'])
        for p in (a,b):files[str(p)]=sha(p)
    assert len(expected)==step;largest=0.;ce_error=0.;counts=Counter()
    with (out/'observed-updates-private.jsonl').open('x') as stream:
        for row,expected_counts in zip(updates,expected):
            assert row['source_counts']==previous[row['local_step']]['source_counts']==expected_counts
            n=sum(expected_counts.values());counts.update(expected_counts)
            loss=sum(row['source_objective'][s]*count*reg['stage1_loss_weights'][s] for s,count in expected_counts.items())/n
            ce=sum(row['source_ce'][s]*count*reg['stage1_loss_weights'][s] for s,count in expected_counts.items())/n
            largest=max(largest,abs(loss-row['loss']));ce_error=max(ce_error,abs(ce-row['ce']))
            stream.write(json.dumps(row,separators=(',',':'))+'\n')
    assert counts==Counter({s:n*epochs for s,n in data['stage1_strata']['train'].items()})
    exposures=reg['train_records']*epochs
    assert sum(counts.values())==updates[-1]['records_seen']==exposures and largest<5e-6 and ce_error<5e-6
    seeds=set(range(reg['curriculum']['curve_seed_start'],reg['curriculum']['curve_seed_start']+2))
    expected_grid={(seed,p,s) for seed in seeds for p in reg['curriculum']['opponent_profiles'] for s in range(4)}
    summaries={};games={};identities={};financial={};reference=ROOT/reg['curriculum']['reference_evaluation_cache']
    for name,folder in [('hard_ce',old/'evaluations'/f'step-{step:06d}'),('max_class',new/'evaluations'/f'step-{step:06d}'),('v8',reference)]:
        for filename in ('matches-private.jsonl','summary.json','protocol.json'):files[str(folder/filename)]=sha(folder/filename)
        all_rows=[json.loads(line) for line in (folder/'matches-private.jsonl').open()]
        stored=read(folder/'summary.json');protocol=read(folder/'protocol.json')
        for key,value in aggregate_complete_matches(all_rows).items():assert stored[key]==value,key
        rows=[r for r in all_rows if r['seed'] in seeds]
        assert len(rows)==16 and {(r['seed'],r['profile'],r['seat']) for r in rows}==expected_grid
        money=Counter();events=Counter()
        for row in rows:
            assert row['completed'] and row['ledger_passed'];seat=row['seat'];ledger=settlement_ledger(row['mjai_log'],row['final_scores'])
            assert ledger['hands']==row['hands'] and row['net_points']==row['final_scores'][seat]-25000
            assert sum(ledger['components'][seat].values())==row['net_points'];money.update(ledger['components'][seat])
            assert sorted(range(4),key=lambda p:(-row['final_scores'][p],p)).index(seat)+1==row['rank']
            assert hand_dealin_count(row['mjai_log'],seat)==row['dealin_hands']
            assert sum(e['type']=='hora' and e.get('actor')==seat for e in row['mjai_log'])==row['wins']
            reached=False;events['hands']+=row['hands'];events['wins']+=row['wins']
            for event in row['mjai_log']:
                if event['type']=='start_kyoku':reached=False
                if event['type']=='reach' and event['actor']==seat:reached=True;events['reach_declarations']+=1
                if event['type']=='hora' and event['actor']!=seat and event['target']==seat:
                    events['ron_events_after_own_reach' if reached else 'ron_events_without_own_reach']+=1
        identity=reg['curriculum']['reference_identity'] if name=='v8' else bundle((old if name=='hard_ce' else new)/'training-run/checkpoints'/f'step-{step:06d}')['identity_sha256']
        assert stored['checkpoint_sha256']==protocol['checkpoint']['identity_sha256']==identity
        assert protocol['profiles']==reg['curriculum']['opponent_profiles'] and protocol['seats']==4
        summaries[name]=aggregate_complete_matches(rows);games[name]=rows;identities[name]=identity
        financial[name]=dict(point_components_per_match={k:v/16 for k,v in money.items()},counts=dict(events))
        assert sum(money.values())/16==summaries[name]['net_points']
    comparisons={}
    for name in ('hard_ce','v8'):
        previous={(r['seed'],r['profile'],r['seat']):r for r in games[name]};comparison={}
        for field,fn in [('net_points',lambda r:r['net_points']),('top2',lambda r:float(r['rank']<=2)),
            ('hand_win',lambda r:r['wins']/r['hands']),('dealin',lambda r:r['dealin_hands']/r['hands']),('fourth',lambda r:float(r['rank']==4))]:
            clusters=[statistics.mean(fn(r)-fn(previous[r['seed'],r['profile'],r['seat']]) for r in games['max_class'] if r['seed']==seed) for seed in sorted(seeds)]
            comparison[field]=dict(gain=statistics.mean(clusters),seed_group_gains=clusters,seed_groups=2,not_independent_strength_proof=True)
        comparisons[name]=comparison
    if export_curves:
        from tools.export_curriculum_riichi_curves_v1 import export
        export(registered,out/'curves',step)
    assert all(sha(p)==h for p,h in files.items())
    keys=('weighted_target_ce','weighted_class_objective','stratum_ce','stratum_accuracy','riichi_by_source','riichi_type_counts')
    result=dict(status='complete_epoch_cached_RI_profiles_and_matched16_reconciled',step=step,
        validation={name:{k:p[k] for k in keys} for name,p in points.items()},source_mechanisms=mechanisms,
        new_source_quality=grade(points['max_class']),complete_prediction_menus_per_model=len(records),
        actual_complete_epochs=epochs,actual_exposures=exposures,actual_source_exposure=dict(counts),
        all_actual_orders_and_source_counts_match_original=True,maximum_objective_reconstruction_error=largest,
        maximum_original_ce_reconstruction_error=ce_error,complete_development_games_checked=48,
        matched16_summaries=summaries,financial_components=financial,checkpoint_identities=identities,
        paired_comparisons=comparisons,logical_seed_groups=2,no_confidence_or_stable_strength_claim=True,
        no_new_model_inference_games_updates_or_threshold_tuning=True,
        no_intermediate_selection_parent_admission_or_final_grid_opened=True,independent_han_fu_scoring=False,
        raw_details_private=True)
    write(out/'summary-private.json',result);write(out/'binding-private.json',dict(files=files,node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']))
    print(json.dumps(dict(status=result['status'],step=step,actual_exposures=exposures,validation=result['validation'],matched16=summaries)),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--registered',type=Path,required=True)
    parser.add_argument('--step',type=int,choices=(152,304,456),required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--no-curves',action='store_true');args=parser.parse_args()
    audit(args.registered.resolve(),args.step,args.out.resolve(),not args.no_curves)
