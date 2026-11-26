"""Publishable real hard-CE curves from audited scalar CSVs only; no new games."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import socket
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha


def export(source,out):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('A bounded scheduler GPU allocation is required')
    out.mkdir(parents=True,exist_ok=False)
    names=('train-curve.csv','validation-curve.csv','evaluation-matched16.csv','evaluation-final64.csv','provenance.json')
    files={str((source/name).resolve()):sha(source/name) for name in names}
    files[str(Path(__file__).resolve())]=sha(__file__)
    provenance=read(source/'provenance.json')
    assert provenance['method']=='scalar_only_registered_basic_repair_curves_v1'
    assert provenance['registered_final_step']==608 and provenance['final64_separate'] is True
    assert provenance['raw_hands_labels_worlds_per_record_predictions_and_weights_exported'] is False
    def rows(name):
        with (source/name).open(newline='') as stream:return list(csv.DictReader(stream))
    train=rows(names[0]);validation=rows(names[1]);evaluation=rows(names[2]);final=rows(names[3])
    steps=lambda values:[int(r['local_step']) for r in values]
    assert steps(train)==list(range(1,609))
    assert steps(validation)==list(range(0,609,76))
    assert steps(evaluation)==[0,152,304,456,608]
    assert len(final)==1 and int(final[0]['matches'])==64 and int(final[0]['seed_groups'])==8
    assert int(final[0]['local_step'])==608
    assert all(math.isclose(float(r['loss']),float(r['ce']),rel_tol=1e-6,abs_tol=1e-6) for r in train)
    assert all(int(r['matches'])==16 and int(r['seed_groups'])==2 for r in evaluation)
    assert all(math.isclose(float(r['reward']),float(r['net_points'])/8000,abs_tol=1e-12) for r in evaluation+final)
    def floats(values,key):
        result=[float(r[key]) for r in values]
        if not all(math.isfinite(v) for v in result):raise ValueError('Missing/nonfinite real scalar')
        return result
    recalls={'accept':[],'decline':[]}
    for row in validation:
        count={name:sum(int(row[source+':'+name]) for source in ('legacy','teacher')) for name in ('tp','fn','tn','fp')}
        assert count['tp']+count['fn']==324 and count['tn']+count['fp']==308
        recalls['accept'].append(count['tp']/324)
        recalls['decline'].append(count['tn']/308)
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.fonttype']='none'
    matplotlib.rcParams['svg.hashsalt']='jev-basic-repair-curves-v1'
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(10,12),sharex=True)
    axes[0].plot(steps(train),floats(train,'loss'),linewidth=.7,alpha=.65,label='Train weighted action CE (each actual update)')
    axes[0].plot(steps(validation),floats(validation,'loss'),marker='o',linewidth=1.3,label='Validation weighted action CE')
    axes[0].set_ylabel('Weighted CE')
    axes[1].plot(steps(validation),recalls['accept'],marker='o',label='Riichi acceptance recall')
    axes[1].plot(steps(validation),recalls['decline'],marker='o',label='Riichi refusal recall')
    axes[1].axhline(.6,color='gray',linestyle=':',label='Registered recall check (not game strength)')
    axes[1].set_ylabel('Recall');axes[1].set_ylim(0,1)
    axes[2].plot(steps(evaluation),floats(evaluation,'reward'),marker='o',label='Observed match reward: net points / 8,000')
    axes[2].axhline(0,color='gray',linestyle=':',linewidth=.7);axes[2].set_ylabel('Match reward')
    for field,label in (('hand_win_rate','Hand win'),('top2_rate','Top two'),('first_rate','First'),('dealin_hand_rate','Deal-in')):
        axes[3].plot(steps(evaluation),floats(evaluation,field),marker='o',label=label)
    axes[3].set_ylabel('Rate');axes[3].set_ylim(0,1);axes[3].set_xlabel('Actual optimizer updates')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
    fig.suptitle('Fresh-base basic repair: hard-action-CE control',fontsize=13)
    fig.text(.5,.006,'Unsmoothed real logs. Game curves: same 16 development matches / 2 seed clusters. Final 64 reported separately.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.02,1,.97))
    for extension in ('svg','png','pdf'):
        options={'metadata':{'Date':None}} if extension=='svg' else {}
        fig.savefig(out/f'basic-repair-curves.{extension}',dpi=170,**options)
    plt.close(fig)
    result=dict(method='audited_scalar_hard_ce_loss_riichi_reward_rate_curves_v1',input_files=files,
        train_points=608,validation_points=9,matched16_game_points=5,final64_drawn_on_curve=False,
        supervised_training_reward=None,game_reward_definition='terminal_raw_net_points_div8000_v1',
        no_smoothing_or_synthetic_observation_points=True,
        note='Straight segments connect observed points only; all game points use the same16-game development grid.',
        no_new_games_or_model_updates=True,raw_private_records_exported=False,
        actual_final64_scalars=final[0],registered_final_only=True,goal_achieved=False,
        output_files={name:sha(out/name) for name in ('basic-repair-curves.svg','basic-repair-curves.png','basic-repair-curves.pdf')})
    if not all(sha(path)==digest for path,digest in files.items()):raise ValueError('Audited scalar source changed')
    with (out/'provenance.json').open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();export(args.source.resolve(),args.out.resolve())
