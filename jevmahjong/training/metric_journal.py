"""Append-only metrics for curriculum figures; no synthetic reward values."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

KINDS={'train','validation','evaluation'}
RATE_FIELDS={'hand_win_rate','first_rate','top2_rate','fourth_rate','dealin_hand_rate'}

class MetricJournal:
    def __init__(self,directory,run_id,stage,stage_offset=0):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.run_id=run_id;self.stage=stage;self.offset=stage_offset

    def append(self,kind,local_step,**metrics):
        if kind not in KINDS or isinstance(local_step,bool) or not isinstance(local_step,int) or local_step<0:
            raise ValueError('Real nonnegative optimizer step and known metric kind required')
        if kind=='train' and metrics.get('reward') is not None:
            raise ValueError('Supervised loss is not observed game reward')
        if kind=='evaluation':
            if not metrics.get('checkpoint_sha256') or metrics.get('matches',0)<=0 or metrics.get('seed_groups',0)<=0:
                raise ValueError('Evaluation must identify weights and actual complete-game denominators')
            if metrics.get('reward') is not None and not metrics.get('reward_definition'):
                raise ValueError('Observed reward needs a fixed definition')
        for key,value in metrics.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):
                if not math.isfinite(value):raise ValueError('Nonfinite metric')
                if key in RATE_FIELDS and not 0<=value<=1:raise ValueError('Rates must be fractions with explicit denominators')
        event=dict(schema='jev-curriculum-metrics-v1',run_id=self.run_id,stage=self.stage,
            local_step=local_step,global_step=self.offset+local_step,
            utc=datetime.now(timezone.utc).isoformat(),**metrics)
        with (self.directory/f'{kind}_metrics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n')
        return event

def aggregate_complete_matches(rows):
    """Equal-weight matches, not pooled hands; raw points remain separate."""
    if not rows or any(not r.get('completed') or r['hands']<=0 or r['rank'] not in (1,2,3,4) for r in rows):
        raise ValueError('Only completed matches with hand denominators are admitted')
    n=len(rows);mean=lambda f:sum(f(r) for r in rows)/n
    return dict(matches=n,seed_groups=len({r['seed'] for r in rows}),
        hand_win_rate=mean(lambda r:r['wins']/r['hands']),
        dealin_hand_rate=mean(lambda r:r['dealin_hands']/r['hands']),
        first_rate=mean(lambda r:float(r['rank']==1)),top2_rate=mean(lambda r:float(r['rank']<=2)),
        fourth_rate=mean(lambda r:float(r['rank']==4)),net_points=mean(lambda r:r['net_points']),
        reward=mean(lambda r:r['net_points']/8000),reward_definition='terminal_raw_net_points_div8000_v1',
        independent_seed_groups_required_for_intervals=True)

def export_csv(directory):
    directory=Path(directory)
    for kind in sorted(KINDS):
        source=directory/f'{kind}_metrics.jsonl'
        if not source.exists():continue
        rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        if not rows:continue
        keys=sorted(set().union(*(set(r) for r in rows)))
        with source.with_suffix('.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)

class LocalTracker:
    """Offline JevAny tracker adapter: observe real post-update metrics only."""
    id='local-journal';project='jev-curriculum';entity=None;url=None
    def __init__(self,journal,name):self.journal=journal;self.name=name;self.config={}
    def log(self,values):
        step=values.get('optimizer_step')
        if 'train/objective' in values:
            self.journal.append('train',step,loss=values['train/objective'],
                ce=values.get('train/ce'),lr=values.get('train/lr'),
                grad_norm=values.get('train/grad_norm_pre_clip'),records_seen=values.get('train/records_seen'),
                reward=None,raw_upstream_metrics=values)
    def finish(self):export_csv(self.journal.directory)
