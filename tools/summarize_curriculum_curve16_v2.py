"""Summarize curve16 progress, with completed endpoint64 reported separately.

Run inside an allocated GPU job. Raw journals and native records stay in runs/;
Only aggregate CSV/Markdown outputs are publication inputs. Original journals
remain separate from the explicitly projected curve16 points.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_development_projection_v1 import source_groups,project_step


def close_number(actual,expected):
    if (type(actual) not in (int,float) or type(expected) not in (int,float)
            or not math.isfinite(actual) or not math.isfinite(expected)
            or not math.isclose(actual,expected,rel_tol=0,abs_tol=1e-10)):
        raise ValueError('Actual journal, summary or native accounting differs')


def summarize(registered,steps,out):
    allocation=resource();registered=Path(registered).resolve();out=Path(out).resolve()
    if (steps!=sorted(set(steps)) or len(steps)<2 or steps[0]!=0
            or any(type(step) is not int or step<0 for step in steps)):
        raise ValueError('Explicit unique increasing development steps beginning at0 required')
    if out.exists() or not out.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Use a new private runs/ output directory; never overwrite evidence')
    reg=read(registered);c=reg['curriculum'];fit=ROOT/c['out']
    if c['curve_seed_groups']!=2 or c['opponent_profiles']!=['efficiency','calling']:
        raise ValueError('Only the registered same16 development grid is supported')
    expected_seeds=list(range(c['curve_seed_start'],c['curve_seed_start']+2))
    if any(2026300000<=seed<2026300032 for seed in expected_seeds):
        raise ValueError('Reserved final seeds cannot enter a development report')
    # Require complete outcomes before reading them, creating outputs, or drawing
    # comparisons. Missing/live points are normal prerequisites, not zero scores.
    for step in steps:
        if read(fit/'evaluations'/f'step-{step:06d}'/'status.json')!={'status':'complete','games':source_groups(c,step)*8}:
            raise ValueError('Wait for the entire original16/64-game development point to complete')
    sys.path.insert(0,reg['source'])
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    metrics={};prefixes=[];files={str(Path(__file__)):sha(__file__),str(registered):sha(registered),
        str(ROOT/'tools/curriculum_development_projection_v1.py'):sha(ROOT/'tools/curriculum_development_projection_v1.py')}
    for kind in ('validation','evaluation'):
        path=fit/'metrics'/(kind+'_metrics.jsonl');raw=path.read_bytes()
        complete=raw[:raw.rfind(b'\n')+1]
        rows=[json.loads(line) for line in complete.splitlines() if line.strip()]
        chosen=[row for row in rows if row['local_step'] in steps]
        if len(chosen)!=len(steps) or {row['local_step'] for row in chosen}!=set(steps):
            raise ValueError('Exactly one actual journal point per selected step required')
        metrics[kind]={row['local_step']:row for row in chosen}
        prefixes.append(dict(path=str(path),complete_prefix_bytes=len(complete),
            complete_prefix_sha256=hashlib.sha256(complete).hexdigest()))
    protocols={};curve_points={};endpoint64={}
    for step in steps:
        folder=fit/'evaluations'/f'step-{step:06d}'
        protocol=read(folder/'protocol.json');summary=read(folder/'summary.json')
        rows=[json.loads(line) for line in (folder/'matches-private.jsonl').read_text().splitlines() if line.strip()]
        curve,full=project_step(c,step,protocol,summary,rows,metrics['evaluation'][step])
        curve_points[step]=curve
        if full is not None:endpoint64[step]=full
        for row in rows:
            if row['completed'] is not True or row['ledger_passed'] is not True:
                raise ValueError('Truncated or unaudited matches cannot enter a report')
            ledger=settlement_ledger(row['mjai_log'],row['final_scores'])
            if (ledger['hands']!=row['hands']
                    or row['wins']!=sum(e['type']=='hora' and e.get('actor')==row['seat'] for e in row['mjai_log'])
                    or row['dealin_hands']!=hand_dealin_count(row['mjai_log'],row['seat'])):
                raise ValueError('Actual hand, win or dealin financial accounting differs')
            close_number(row['net_points'],row['final_scores'][row['seat']]-25000)
        aggregate=aggregate_complete_matches(rows)
        for key,value in aggregate.items():
            if type(value) in (int,float):close_number(value,summary[key])
            elif value!=summary[key]:raise ValueError('Actual game aggregate definition differs')
        protocols[step]=protocol
        for name in ('protocol.json','summary.json','matches-private.jsonl','status.json'):
            files[str(folder/name)]=sha(folder/name)
        model=protocol['checkpoint']
        for name,expected_digest in model['files'].items():
            path=Path(model['directory'])/name
            if sha(path)!=expected_digest:raise ValueError('Actual evaluated parameter bundle changed')
            files[str(path)]=expected_digest
    for step in steps[1:]:
        if (any(protocols[0][key]!=protocols[step][key] for key in ('profiles','seats','kind'))
                or protocols[step]['seeds'][:2]!=expected_seeds
                or {k:v for k,v in protocols[0]['policy'].items() if k!='parameter_identity'}
                    !={k:v for k,v in protocols[step]['policy'].items() if k!='parameter_identity'}):
            raise ValueError('A serving-policy or grid change cannot masquerade as training progress')
    ready_path=(ROOT/c['data_ready']).resolve();ready=read(ready_path);files[str(ready_path)]=sha(ready_path)
    for step in steps:
        if metrics['validation'][step]['validation_data_sha256']!=ready['files']['validation']['sha256']:
            raise ValueError('Every selected validation point requires the identical registered dataset')
    items=[('validation','loss','Validation objective'),('validation','ce','Validation weighted target CE')]
    for key,label in (('natural_target_ce','Validation natural target CE'),
            ('new_route_objective','New-route weighted objective contribution')):
        present=[key in metrics['validation'][step] for step in steps]
        if any(present) and not all(present):raise ValueError('Do not mix validation schemas')
        if all(present):items.append(('validation',key,label))
    items.extend(('evaluation',key,label) for key,label in (
        ('top2_rate','Top-two rate'),('net_points','Net points per match'),
        ('hand_win_rate','Hand win rate'),('dealin_hand_rate','Dealin hand rate'),
        ('first_rate','First-place rate'),('fourth_rate','Fourth-place rate'),('reward','Match reward (net /8000)')))
    table=[];long_rows=[]
    for kind,key,label in items:
        values=[(curve_points if kind=='evaluation' else metrics[kind])[step][key] for step in steps]
        for value in values:close_number(value,value)
        table.append(dict(kind=kind,metric=key,label=label,values=dict(zip(steps,values)),
            change_from_initial=values[-1]-values[0],change_from_previous=values[-1]-values[-2]))
        for index,(step,value) in enumerate(zip(steps,values)):
            long_rows.append(dict(kind=kind,metric=key,label=label,step=step,value=value,
                change_from_initial=value-values[0],change_from_previous=0 if index==0 else value-values[index-1]))
    out.mkdir(parents=True)
    write(out/'selected-journal-points-private.json',metrics);write(out/'journal-prefix-receipts-private.json',prefixes)
    write(out/'projected-curve16-points-private.json',curve_points)
    write(out/'original-endpoint64-aggregates-private.json',endpoint64)
    write(out/'aggregate-comparison-private.json',table)
    with (out/'aggregate-comparison.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(long_rows[0]));writer.writeheader();writer.writerows(long_rows)
    lines=['# Same-grid curriculum development progress','',
        f"All points use the same {ready['files']['validation']['records']:,} validation records and16 complete East matches "
        '(two consumed logical seeds, two opponent profiles, four seats). Parameters changed; the execution policy and grid stayed fixed.','',
        '| Metric | '+' | '.join('Step'+str(step) for step in steps)+f' | Change {steps[-2]}→{steps[-1]} |',
        '| --- | '+' | '.join('---:' for _ in steps)+' | ---: |']
    for row in table:
        rate=row['metric'].endswith('_rate');scale=100 if rate else 1
        values=[f"{row['values'][step]*scale:.5f}"+('%' if rate else '') for step in steps]
        delta=f"{row['change_from_previous']*scale:+.5f}"+(' pp' if rate else '')
        lines.append('| '+row['label']+' | '+' | '.join(values)+' | '+delta+' |')
    if endpoint64:
        lines+=['','## Separate full endpoint64 results','',
            'The curve above uses only the same first two registered logical seeds at every step. '
            'The endpoint below uses all eight registered development seed groups and is not joined to that curve.','',
            '| Step | Matches | Top two | Hand win | Deal-in | Net points / match | Reward |',
            '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for step,row in sorted(endpoint64.items()):
            lines.append(f"| {step} | {row['matches']} | {row['top2_rate']:.2%} | {row['hand_win_rate']:.2%} | "
                f"{row['dealin_hand_rate']:.2%} | {row['net_points']:.2f} | {row['reward']:.5f} |")
        with (out/'endpoint64-aggregate.csv').open('x',newline='') as stream:
            fields=['step',*next(iter(endpoint64.values()))]
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            writer.writerows(dict(step=step,**row) for step,row in sorted(endpoint64.items()))
    lines+=['','This consumed development grid has only two seed groups. It does not establish independent strength, '
        'rare-yaku mastery or superiority to frozen v8. No confidence interval or checkpoint-selection claim is made here. '
        'New-route objective is a fixed-weight validation loss contribution; match reward is net points divided by8000. '
        'Natural CE is retained to disclose tradeoffs.','',
        'This report reads completed artifacts only. It generates no games, changes no training recipe or serving policy, '
        'and does not open final confirmation or replace production weights. Raw matches and validation menus remain private.','']
    (out/'development-summary.md').write_text('\n'.join(lines))
    summary=dict(status='actual_same16_development_points_and_full_validation_summarized',
        actual_steps=steps,validation_records=ready['files']['validation']['records'],matches_per_point=16,
        consumed_seed_groups=2,same_grid_and_execution_policy=True,native_financial_ledgers_reconstructed=True,
        original_source_matches={step:source_groups(c,step)*8 for step in steps},
        endpoint64_steps=sorted(endpoint64),endpoint64_not_joined_to_curve=True,
        independent_strength_proved=False,final_confirmation_opened=False,new_games=0,model_loaded=False,student_updates=0,
        training_recipe_unchanged=True,production_unchanged=True,allocation=allocation)
    write(out/'summary-private.json',summary)
    for path in out.iterdir():
        if path.is_file():files[str(path)]=sha(path)
    write(out/'completion-binding-private.json',dict(files=files))
    print(json.dumps(dict(status=summary['status'],steps=steps,comparison=table)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registered',type=Path,required=True)
    parser.add_argument('--steps',type=int,nargs='+',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();summarize(args.registered,args.steps,args.out)
