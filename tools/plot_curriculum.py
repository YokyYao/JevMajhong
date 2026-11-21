"""Render real curriculum logs to standalone PNG/PDF and export the raw CSV."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevmahjong.training.metric_journal import export_csv

def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory);export_csv(directory)
    def read(kind):
        path=directory/f'{kind}_metrics.jsonl'
        return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
    train=read('train');validation=read('validation');evaluation=read('evaluation')
    if not train:raise ValueError('No real optimizer metrics; refusing to invent a training curve')
    fig,axes=plt.subplots(3,1,figsize=(9,11),sharex=True)
    # Plot per-stage segments: different losses/rewards are not silently joined.
    for stage in dict.fromkeys(r['stage'] for r in train+validation+evaluation):
        for rows,key,ax,label in [(train,'loss',axes[0],'train loss'),
                (validation,'loss',axes[0],'validation loss'),
                (evaluation,'reward',axes[1],'evaluation reward'),
                (evaluation,'hand_win_rate',axes[2],'hand-win'),
                (evaluation,'first_rate',axes[2],'first'),
                (evaluation,'top2_rate',axes[2],'top-two')]:
            chosen=[r for r in rows if r['stage']==stage and r.get(key) is not None]
            definitions={r.get('reward_definition') for r in chosen} if key=='reward' else {key}
            for definition in definitions:
                points=[r for r in chosen if key!='reward' or r.get('reward_definition')==definition]
                if not points:continue
                axes_label=f'{stage}: {label}'+(f' ({definition})' if key=='reward' else '')
                ax.plot([r['global_step'] for r in points],[r[key] for r in points],
                    marker='.' if rows is not train else None,linewidth=.9,label=axes_label)
        steps=[r['global_step'] for r in train if r['stage']==stage]
        if steps:
            for ax in axes:ax.axvline(min(steps),color='grey',linestyle=':',alpha=.4)
    for ax,label in zip(axes,['Loss (stage-specific)','Observed evaluation reward','Rate (fraction)']):
        ax.set_ylabel(label);ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    axes[-1].set_xlabel('Cumulative actual optimizer updates')
    axes[-1].set_ylim(0,1)
    fig.suptitle('Raw curriculum metrics — development evaluations, not independent test results')
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(directory/f'curriculum-curves.{suffix}',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metrics-dir',type=Path,required=True)
    plot(p.parse_args().metrics_dir)
