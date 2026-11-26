"""Saved-prediction decomposition and logged clipping coefficient accounting.

No model, new labels, parameter gradients, optimizer updates or games. The
probability identity is descriptive; components are not independently
intervened on at the parameter level. Clip coefficients are not gradient norms
or Adam update attributions.
"""
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write


def main():
    allocation=resource();parent=ROOT/'runs/curriculum-progression-parent-retention-v1'
    out=parent/'ri-margin-clip-diagnosis-v1'
    if out.exists():raise ValueError('Never replace a diagnostic attempt')
    out.mkdir()
    registration=parent/'fit-plan/registered.json';reg=read(registration)
    receipt=read(reg['training_config']['data_receipt']);validation=Path(receipt['files']['validation']['path'])
    if sha(validation)!=receipt['files']['validation']['sha256']:raise ValueError('Fixed validation changed')
    truth={};all_ids=set();files={str(registration):sha(registration),str(validation):sha(validation),str(Path(__file__).resolve()):sha(__file__)}
    with validation.open() as stream:
        for line in stream:
            row=json.loads(line);identifier=row['_meta']['id'];all_ids.add(identifier);q=row['questions']['action']
            if q['src'].endswith((':riichi_no',':riichi_yes')):
                reaches=[k for k,v in q['criteria'].items() if v['type']=='reach']
                if len(reaches)!=1 or {v['type'] for v in q['criteria'].values()}!={'reach','dahai'}:
                    raise ValueError('One reach and full discard menu required')
                truth[identifier]=(q['src'],reaches[0],set(q['criteria']))
    if len(all_ids)!=23688 or len(truth)!=632:raise ValueError('Original complete validation and632 RI menus required')
    points={};max_identity_error=0.
    for step in (0,600,1200):
        path=parent/'stage3-fit/training-run/training_eval'/f'step-{step:06d}'/'development/predictions.jsonl'
        files[str(path)]=sha(path);values={};seen=set()
        with path.open() as stream:
            for line in stream:
                raw=json.loads(line);identifier=raw['id']
                if identifier in seen:raise ValueError('Duplicate prediction ID')
                seen.add(identifier)
                if identifier not in truth:continue
                source,reach,keys=truth[identifier];row=raw['rows'][0]
                if row['task']!=source or set(row['keys'])!=keys:raise ValueError('Actual prediction alignment changed')
                p=row['p'];r=p[row['keys'].index(reach)];d=[v for k,v in zip(row['keys'],p) if k!=reach]
                if not 0<r<1 or not max(d)>0 or not sum(d)>0:raise ValueError('Finite nondegenerate class probabilities required')
                maximum=math.log(r/max(d));gap=math.log(sum(d)/max(d));full=math.log(r/sum(d))
                error=abs(full-(maximum-gap));max_identity_error=max(max_identity_error,error)
                if error>1e-12:raise ValueError('RI probability-ratio identity failed')
                values[identifier]={'reach_probability':r,'reach_to_max_discard':maximum,
                    'discard_logsumexp_minus_max':gap,'full_class_margin':full}
        if seen!=all_ids or set(values)!=set(truth):raise ValueError('Incomplete saved comparison')
        points[step]=values
    strata={};details={}
    for source in sorted({v[0] for v in truth.values()}):
        ids=[i for i,v in truth.items() if v[0]==source]
        means={str(step):{key:statistics.mean(points[step][i][key] for i in ids)
            for key in next(iter(points[step].values()))} for step in points}
        changes={key:means['1200'][key]-means['0'][key] for key in means['0']}
        flips=[i for i in ids if source.endswith(':riichi_no') and points[0][i]['reach_probability']<.5<=points[1200][i]['reach_probability']]
        maximum_only=sum(points[0][i]['full_class_margin']+
            points[1200][i]['reach_to_max_discard']-points[0][i]['reach_to_max_discard']>=0 for i in flips)
        gap_only=sum(points[0][i]['full_class_margin']-
            (points[1200][i]['discard_logsumexp_minus_max']-points[0][i]['discard_logsumexp_minus_max'])>=0 for i in flips)
        strata[source]={'n':len(ids),'means':means,'changes0_to1200':changes,
            'new_false_accepts':len(flips),'maximum_margin_change_alone_crosses_threshold':maximum_only,
            'discard_gap_change_alone_crosses_threshold':gap_only,
            'counterfactuals_are_arithmetic_not_parameter_interventions':True}
        details[source]={i:{str(step):points[step][i] for step in points} for i in ids}
    import torch
    torch.set_num_threads(1)
    journal=parent/'milestone1200-plots-v1/train-journal-through1200-private.jsonl'
    with journal.open() as stream:training=[json.loads(line) for line in stream]
    if [r['local_step'] for r in training]!=list(range(1,1201)):raise ValueError('Actual completed first epoch required')
    files[str(journal)]=sha(journal);weights=reg['course_loss_weights'];before=Counter();after=Counter();scales=[]
    present={s:[] for s in strata}
    for row in training:
        norm=row['grad_norm'];n=sum(row['source_counts'].values())
        if not math.isfinite(norm) or norm<=0:raise ValueError('Finite positive actual preclip norm required')
        dummy=torch.nn.Parameter(torch.zeros(1,dtype=torch.float64));dummy.grad=torch.tensor([norm],dtype=torch.float64)
        torch.nn.utils.clip_grad_norm_([dummy],1.)
        scale=float(dummy.grad[0])/norm;scales.append(scale)
        for source,count in row['source_counts'].items():
            coefficient=count*weights[source]/n;before[source]+=coefficient;after[source]+=coefficient*scale
        for source in strata:
            if row['source_counts'].get(source,0):present[source].append(scale)
    btotal=sum(before.values());atotal=sum(after.values())
    clipping={s:{'nominal_normalized_coefficient_share':before[s]/btotal,
        'after_global_clip_normalized_coefficient_share':after[s]/atotal,
        'nominal_coefficient_weighted_mean_clip_scale':after[s]/before[s],
        'mean_clip_scale_when_source_present':statistics.mean(present[s]),
        'updates_with_source':len(present[s])} for s in strata}
    pairs={}
    for origin in ('legacy','teacher'):
        no='curriculum2:basic:'+origin+':riichi_no';yes='curriculum2:basic:'+origin+':riichi_yes'
        pairs[origin]={'nominal_decline_to_accept_coefficient_ratio':before[no]/before[yes],
            'after_clip_decline_to_accept_coefficient_ratio':after[no]/after[yes]}
    report={'status':'saved_RI_margin_identity_and_logged_clip_coefficient_diagnosis_complete',
        'RI_menus':len(truth),'validation_points':[0,600,1200],'strata':strata,
        'maximum_probability_identity_error':max_identity_error,
        'actual_train_updates':len(training),'clipped_updates':sum(v<1 for v in scales),
        'mean_global_clip_scale':statistics.mean(scales),'source_clip_coefficients':clipping,
        'decline_accept_clip_ratios':pairs,'clip_coefficients_estimated_from_logged_preclip_norms':True,
        'gradient_directions_Adam_moments_and_weight_decay_not_attributed':True,
        'model_loaded':False,'student_updates':0,'new_games':0,'validation_not_used_as_TRAIN':True,
        'live_trial_changed':False,'allocation':allocation}
    write(out/'details-private.json',details);write(out/'summary-private.json',report)
    for p in (out/'details-private.json',out/'summary-private.json',Path(torch.nn.utils.clip_grad_norm_.__code__.co_filename)):
        files[str(p)]=sha(p)
    write(out/'completion-binding-private.json',{'files':files})
    print(json.dumps({k:v for k,v in report.items() if k!='allocation'}),flush=True)


if __name__=='__main__':main()
