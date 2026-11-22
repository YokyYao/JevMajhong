"""Export aggregate-only fixed RI diagnostics and a real standalone figure."""
import argparse
import csv
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha


def export(audit,out):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-','login'))):raise RuntimeError('Bounded scheduler GPU required')
    value=read(audit/'summary-private.json');bound=read(audit/'binding-private.json')['files']
    if (value['status']!='fixed_cached_riichi_mechanisms_and_actual_update_mass_audited'
        or value['steps']!=[0,76,152,228] or not all(sha(p)==h for p,h in bound.items())):
        raise ValueError('Complete unchanged fixed diagnostic evidence required')
    inputs={str(p.resolve()):sha(p) for p in (Path(__file__),audit/'summary-private.json',audit/'binding-private.json')}
    rows=[]
    for step in value['steps']:
        snapshot=value['snapshots'][str(step)]
        for source in ('legacy','teacher','pooled'):
            point=snapshot['pooled'] if source=='pooled' else snapshot['sources'][source]
            raw=point['confusion']['raw_argmax'];fixed=point['confusion']['fixed_full_class_0_5']
            yes=raw['tp']+raw['fn'];no=raw['tn']+raw['fp'];d=point['fixed_decoder_disagreements']
            row=dict(step=step,source=source,windows=point['windows'],positive= yes,negative=no,
                raw_acceptance=raw['tp']/yes,raw_correct_refusal=raw['tn']/no,
                fixed05_acceptance=fixed['tp']/yes,fixed05_correct_refusal=fixed['tn']/no,
                fixed_decoder_disagreements=d.get('total',0),disagreements_teacher_yes=d.get('teacher_yes',0),
                disagreements_teacher_no=d.get('teacher_no',0),
                mean_p_riichi_teacher_yes=point['cohorts']['teacher_yes']['p_riichi']['mean'],
                mean_p_riichi_teacher_no=point['cohorts']['teacher_no']['p_riichi']['mean'])
            for k in ('class_probability_auc','argmax_logit_margin_auc','class_brier','mean_class_nll',
                'mean_conditional_discard_nll','mean_normalized_action_nll','max_nll_decomposition_error','class_ece10'):row[k]=point[k]
            rows.append(row)
    intervals=[]
    for label,point in value['actual_loss_weight_intervals'].items():
        fraction=point['per_update_riichi_positive_fraction']
        intervals.append(dict(interval=label,first_step=point['first_step'],last_step=point['last_step'],updates=point['updates'],
            positive_share=point['aggregate_riichi_positive_fraction'],minimum=fraction['minimum'],maximum=fraction['maximum'],
            median=fraction['median'],p10=fraction['p10'],p90=fraction['p90'],
            below025=point['steps_positive_fraction_below_0_25'],above075=point['steps_positive_fraction_above_0_75'],
            positive_weighted_ce=point['positive_weighted_ce'],negative_weighted_ce=point['negative_weighted_ce']))
    out.mkdir(parents=True,exist_ok=False)
    for name,records in (('riichi-snapshot-metrics.csv',rows),('riichi-weight-intervals.csv',intervals)):
        with (out/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors={'legacy':'#2864b0','teacher':'#cf641b'}
    for source in ('legacy','teacher'):
        selected=[r for r in rows if r['source']==source];x=[r['step'] for r in selected];color=colors[source]
        for metric,style,label in (('raw_acceptance','-','accept'),('raw_correct_refusal','--','refuse')):
            axes[0,0].plot(x,[r[metric] for r in selected],style,marker='o',color=color,label=source+' '+label)
        for metric,style,label in (('class_probability_auc','-','class P'),('argmax_logit_margin_auc','--','margin')):
            axes[0,1].plot(x,[r[metric] for r in selected],style,marker='o',color=color,label=source+' '+label)
        for metric,style,label in (('mean_class_nll','-','class'),('mean_conditional_discard_nll','--','discard')):
            axes[1,0].plot(x,[r[metric] for r in selected],style,marker='o',color=color,label=source+' '+label)
    half=[r for r in intervals if not r['interval'].startswith('full_epoch')]
    x=[(r['first_step']+r['last_step'])/2 for r in half];y=[r['positive_share'] for r in half]
    axes[1,1].errorbar(x,y,yerr=[[v-r['minimum'] for v,r in zip(y,half)],
        [r['maximum']-v for v,r in zip(y,half)]],fmt='o-',capsize=5,color='#435a44',label='aggregate share; min/max per update')
    axes[1,1].axhline(.5,color='gray',linestyle='--',label='registered epoch share 0.5')
    titles=['Actual raw RI class recall','Ranking against held-out imitation labels',
        'RI action NLL = class NLL + conditional discard NLL','Actual positive RI loss-weight share (76-step intervals)']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Optimizer step');ax.grid(alpha=.2);ax.legend(fontsize=7)
    for ax in (axes[0,0],axes[0,1],axes[1,1]):ax.set_ylim(0,1)
    axes[1,0].set_ylabel('Natural mean NLL per RI window')
    fig.suptitle('Saved basic-repair snapshots: real development diagnostics, no strength claim',fontsize=12)
    for suffix in ('png','svg','pdf'):
        options={'metadata':{'Date':None}} if suffix=='svg' else {}
        fig.savefig(out/f'riichi-mechanisms.{suffix}',dpi=160,**options)
    plt.close(fig)
    if not all(sha(p)==h for p,h in inputs.items()) or not all(sha(p)==h for p,h in bound.items()):raise ValueError('Diagnostic source changed during export')
    provenance=dict(method='aggregate_only_fixed_riichi_mechanisms_export_v1',source_summary_sha256=inputs[str((audit/'summary-private.json').resolve())],
        source_binding_sha256=inputs[str((audit/'binding-private.json').resolve())],export_code_sha256=sha(__file__),
        steps=value['steps'],published_scalar_rows=len(rows),actual_updates_checked=228,
        no_case_ids_groups_hands_labels_predictions_or_per_update_weights_exported=True,
        measured_loss_weights_are_not_gradients=True,retrospective_development_diagnostic_not_causal_ablation=True,
        no_policy_or_training_change=True,goal_achieved=False,outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    with (out/'provenance.json').open('x') as f:json.dump(provenance,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='actual_aggregate_riichi_metrics_and_figure_exported',snapshots=rows,intervals=intervals)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();export(a.audit.resolve(),a.out.resolve())
