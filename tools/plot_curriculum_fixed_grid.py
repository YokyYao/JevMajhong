"""Plot matched16 development games; keep final64 results separate."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.plot_curriculum import plot

def aligned_record(row):
    result=copy.deepcopy(row)
    if row['matches']==64:
        subset=row.get('curve_grid_subset')
        if not subset or subset['matches']!=16 or subset['seed_groups']!=2:
            raise ValueError('Final64 record must provide its verified matching16 subset')
        result.update(subset);result['original_matches']=64;result['grid']='curve16_subset_of_final64'
    elif row['matches']!=16 or row['seed_groups']!=2:
        raise ValueError('Unknown development grid; refuse to connect incomparable samples')
    return result

def complete_rows(path):
    if not path.exists():return []
    raw=path.read_text();lines=raw.splitlines()
    if raw and not raw.endswith('\n'):lines=lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]

def render(directory):
    directory=Path(directory);evaluation=complete_rows(directory/'evaluation_metrics.jsonl')
    aligned=[aligned_record(row) for row in evaluation]
    with tempfile.TemporaryDirectory(prefix='.matched-grid-',dir=directory) as temp:
        stage=Path(temp)
        for kind in ('train','validation'):
            rows=complete_rows(directory/f'{kind}_metrics.jsonl')
            (stage/f'{kind}_metrics.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        (stage/'evaluation_metrics.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in aligned))
        plot(stage)
        for ext in ('png','pdf'):shutil.copy2(stage/f'curriculum-curves.{ext}',directory/f'curriculum-curves.{ext}')
        shutil.copy2(stage/'evaluation_metrics.csv',directory/'evaluation_metrics-fixed-grid.csv')
    fields=('local_step','global_step','stage','matches','seed_groups','hand_win_rate','first_rate',
            'top2_rate','fourth_rate','dealin_hand_rate','net_points','reward','checkpoint_sha256')
    with (directory/'evaluation-stage-end64.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        writer.writerows(r for r in evaluation if r['matches']==64)
    evidence=dict(method='matched16_development_grid_v1',process_curve_matches=16,process_curve_seed_groups=2,
        original_final64_results_unchanged=True,new_policy_games=0,
        curve_steps=[r['local_step'] for r in aligned],final64_steps=[r['local_step'] for r in evaluation if r['matches']==64],
        source_evaluation_sha256=hashlib.sha256((directory/'evaluation_metrics.jsonl').read_bytes()).hexdigest(),
        note='Same16-game subset for all curve points. Final64 has its own CSV/report; neither is independent confirmation.')
    (directory/'plot-data-provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--metrics-dir',type=Path,required=True);render(p.parse_args().metrics_dir)
