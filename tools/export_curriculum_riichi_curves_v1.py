"""Export real RI-class-loss snapshots with minimal public scalar columns.

The complete captured journals remain under snapshots-private, outside public.
No language model, game simulation, smoothing or invented rewards are used.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_riichi_control import METHOD,require_registered
from tools.plot_curriculum_fixed_grid import aligned_record

TRAIN_FIELDS=('local_step','loss','ce','lr','head_lr_derived','records_seen')
GAME_FIELDS=('matches','seed_groups','hand_win_rate','dealin_hand_rate','first_rate',
             'top2_rate','fourth_rate','net_points','reward')


def finite(value):
    if type(value) not in (float,int) or not math.isfinite(value):raise ValueError('Finite real scalar required')
    return value


def check(train,validation,games,through_step,final=False):
    if not train or [r['local_step'] for r in train]!=list(range(1,through_step+1)):
        raise ValueError('Every actual update through the snapshot is required')
    if not 1<=through_step<=608:raise ValueError('Snapshot is outside the registered budget')
    if not validation or [r['local_step'] for r in validation]!=list(range(0,validation[-1]['local_step']+1,76)):
        raise ValueError('Validation must be an unbroken actual76-step prefix')
    if validation[-1]['local_step']>through_step:raise ValueError('Validation beyond the snapshot')
    stages={r['stage'] for r in train+validation+games}
    if stages!={'stage1-riichi-class-control-v1'}:raise ValueError('Mixed or wrong course stages')
    for row in train+validation:
        if row['loss_method']!=METHOD:raise ValueError('Wrong objective; hard CE must not be relabelled')
        finite(row['loss']);finite(row['ce'])
    for row in train:
        if row.get('reward') is not None or row.get('ce_is_original_hard_action_loss') is not True:
            raise ValueError('Supervised reward must remain null and common CE explicit')
        for key in TRAIN_FIELDS:finite(row[key])
    for row in validation:
        finite(row['natural_loss'])
        for source,yes,no in [('legacy',37,70),('teacher',287,238)]:
            counts=row['riichi_by_source'][source]
            if any(type(counts.get(k,0)) is not int or counts.get(k,0)<0 for k in ('tp','fn','tn','fp')):
                raise ValueError('Actual nonnegative confusion counts required')
            if counts.get('tp',0)+counts.get('fn',0)!=yes or counts.get('tn',0)+counts.get('fp',0)!=no:
                raise ValueError('RI source denominator changed')
    if not games or [r['local_step'] for r in games]!=list(range(0,games[-1]['local_step']+1,152)):
        raise ValueError('Games must be the observed152-step prefix, without interpolation')
    if games[-1]['local_step']>validation[-1]['local_step']:raise ValueError('Game point beyond validation')
    matched=[];final64=[]
    for row in games:
        if row.get('reward_definition')!='terminal_raw_net_points_div8000_v1':raise ValueError('Game reward definition changed')
        if row['local_step']==608:
            if row['matches']!=64 or row['seed_groups']!=8:raise ValueError('Final requires the full64 grid')
            final64.append(row)
        elif row['matches']!=16 or row['seed_groups']!=2:raise ValueError('Process game grid changed')
        point=aligned_record(row);matched.append(point)
        for value in (row,point):
            for key in GAME_FIELDS:finite(value[key])
            if not math.isclose(value['reward'],value['net_points']/8000,abs_tol=1e-12):
                raise ValueError('Reward/net mismatch')
    if final and (through_step!=608 or len(validation)!=9 or len(matched)!=5 or len(final64)!=1):
        raise ValueError('Final export requires all actual608/9/5/final64 points')
    return matched,final64


def export(registered,out,through_step=None,require_final=False):
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
        or socket.gethostname().startswith(('lo-','login'))):raise RuntimeError('Scheduled GPU required')
    reg=read(registered);require_registered(reg)
    for path,digest in reg['inputs'].items():
        if sha(path)!=digest:raise ValueError('Frozen registered source changed')
    metrics=ROOT/reg['curriculum']['out']/'metrics'
    out.mkdir(parents=True,exist_ok=False);private=out/'snapshots-private';private.mkdir()
    public=out/'public';public.mkdir();captured={};raw_hashes={}
    for kind in ('train','validation','evaluation'):
        path=metrics/f'{kind}_metrics.jsonl';raw=path.read_bytes()
        # Ignore an incomplete final line only. Snapshot exact bytes privately;
        # the running journal is expected to grow and is never rebound as static.
        (private/path.name).write_bytes(raw);raw_hashes[path.name]=hashlib.sha256(raw).hexdigest()
        lines=raw.splitlines(keepends=True)
        captured[kind]=[json.loads(line) for line in lines if line.endswith(b'\n')]
    if not captured['train']:raise ValueError('No real training updates')
    step=through_step if through_step is not None else captured['train'][-1]['local_step']
    select=lambda kind:[r for r in captured[kind] if r['local_step']<=step]
    train,validation,games=(select(kind) for kind in ('train','validation','evaluation'))
    matched,final64=check(train,validation,games,step,require_final)
    evidence=None
    if require_final:
        evidence_path=(ROOT/reg['curriculum']['out']).parent/'final-evidence-private.json'
        evidence=read(evidence_path)
        if (evidence.get('status')!='stage1_final_evidence_audit_passed'
            or evidence.get('optimizer_steps')!=608 or evidence.get('real_records_seen')!=57140
            or evidence.get('final_summary',{}).get('matches')!=64):
            raise ValueError('Actual independent full final evidence required')
        raw_hashes['final-evidence-private.json']=sha(evidence_path)
    def csv_file(name,fields,rows):
        with (public/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    csv_file('train-curve.csv',TRAIN_FIELDS,train)
    val_fields=('local_step','loss','ce','natural_loss',
        *[source+':'+key for source in ('legacy','teacher') for key in ('tp','fn','tn','fp')])
    val=[]
    for row in validation:
        point={key:row[key] for key in ('local_step','loss','ce','natural_loss')}
        for source in ('legacy','teacher'):
            for key in ('tp','fn','tn','fp'):point[source+':'+key]=row['riichi_by_source'][source].get(key,0)
        val.append(point)
    csv_file('validation-curve.csv',val_fields,val)
    csv_file('evaluation-matched16.csv',('local_step',*GAME_FIELDS),matched)
    csv_file('evaluation-final64.csv',('local_step',*GAME_FIELDS),final64)
    import matplotlib
    matplotlib.use('Agg');matplotlib.rcParams['svg.fonttype']='none'
    matplotlib.rcParams['svg.hashsalt']='jev-riichi-class-curves-v1'
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(10,12),sharex=True)
    for rows,label,marker in [(train,'Train',None),(validation,'Validation','o')]:
        x=[r['local_step'] for r in rows]
        for key,description,style in [('loss','RI class objective','-'),('ce','Original action CE','--')]:
            axes[0].plot(x,[r[key] for r in rows],style,marker=marker,linewidth=.8,
                alpha=.7 if rows is train else 1,label=label+': '+description)
    axes[0].set_ylabel('Weighted loss / CE')
    for source in ('legacy','teacher'):
        for numerator,denominator,label,style in [('tp',('tp','fn'),'accept','-'),('tn',('tn','fp'),'decline','--')]:
            values=[]
            for row in validation:
                counts=row['riichi_by_source'][source];values.append(counts.get(numerator,0)/sum(counts.get(k,0) for k in denominator))
            axes[1].plot([r['local_step'] for r in validation],values,style,marker='o',label=source+' RI '+label)
    axes[1].axhline(.6,color='gray',linestyle=':',label='Registered source recall check')
    axes[1].set_ylabel('Recall');axes[1].set_ylim(0,1)
    x=[r['local_step'] for r in matched]
    axes[2].plot(x,[r['reward'] for r in matched],marker='o',label='Observed match reward: net points / 8,000')
    axes[2].axhline(0,color='gray',linestyle=':',linewidth=.7);axes[2].set_ylabel('Match reward')
    for key,label in [('hand_win_rate','Hand win'),('top2_rate','Top two'),('first_rate','First'),('dealin_hand_rate','Deal-in')]:
        axes[3].plot(x,[r[key] for r in matched],marker='o',label=label)
    axes[3].set_ylabel('Rate');axes[3].set_ylim(0,1);axes[3].set_xlabel('Actual optimizer updates')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
    state='audited final' if require_final else 'unfinished development snapshot'
    fig.suptitle('Fresh RI class-loss control: '+state,fontsize=13)
    fig.text(.5,.006,f'Actual train through{step}; validation through{validation[-1]["local_step"]}; games through{matched[-1]["local_step"]}. Same16 games; final64 separate. No smoothing.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.02,1,.97))
    for ext in ('svg','png','pdf'):
        kwargs={'metadata':{'Date':None}} if ext=='svg' else {}
        fig.savefig(public/f'riichi-class-curves.{ext}',dpi=170,**kwargs)
    plt.close(fig)
    result=dict(method='real_riichi_class_curves_minimal_public_scalars_v1',
        registration_sha256=sha(registered),export_source_sha256=sha(__file__),captured_journal_sha256=raw_hashes,
        actual_train_through=step,actual_validation_steps=[r['local_step'] for r in validation],
        actual_matched16_steps=[r['local_step'] for r in matched],actual_final64_steps=[r['local_step'] for r in final64],
        registered_final_step=608,independent_full_final_audit_checked=require_final,
        loss_is_RI_class_objective=True,ce_is_original_action_CE=True,supervised_reward=None,
        game_reward_definition='terminal_raw_net_points_div8000_v1',
        snapshot_may_lag_live_training=True,source_journals_captured_privately=True,
        no_gradient_source_per_update_diagnostics_raw_hands_or_weights_in_public=True,
        no_new_inference_games_updates_smoothing_or_synthetic_observations=True,goal_achieved=False,
        output_files={p.name:sha(p) for p in public.iterdir()})
    with (public/'provenance.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--registered',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--through-step',type=int)
    parser.add_argument('--require-final',action='store_true');args=parser.parse_args()
    export(args.registered.resolve(),args.out.resolve(),args.through_step,args.require_final)
