"""Describe kan-label coverage and cached decisions, without model inference.

Soft target support is explicit: a legal but unlabelled kan is not a negative
label. Source preference differences do not establish contradictory labels for
the same state, or prove that a particular kan is profitable.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_complete_legal_class_policy_v1 import select
from tools.curriculum_transfer import read,sha

OPS=('ankan','daiminkan','kakan')


def label_kind(q,op):
    criteria=q['criteria'];target=q.get('target')
    if target is None:return 'hard_positive' if criteria[q['label']]['type']==op else 'hard_negative'
    if not target or not set(target)<=set(criteria):raise ValueError('Explicit supported soft target required')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in target.values()):
        raise ValueError('Finite nonnegative soft target values required')
    own=[v for k,v in target.items() if criteria[k]['type']==op]
    other=[v for k,v in target.items() if criteria[k]['type']!=op]
    if not own:return 'soft_operation_unlabelled'
    if not other:return 'soft_no_other_labelled'
    a,b=max(own),max(other)
    return 'soft_preferred' if a>b+1e-10 else 'soft_other_preferred' if b>a+1e-10 else 'soft_tied'


def audit(comparison,out):
    allocation=resource();comparison=Path(comparison).resolve();out=Path(out).resolve()
    if out.exists() or not out.is_relative_to((ROOT/'runs').resolve()):raise ValueError('New private output required')
    binding_path=comparison/'completion-binding-private.json';bound=read(binding_path)['files']
    files={str(Path(__file__)):sha(__file__),str(binding_path):sha(binding_path),
        str(ROOT/'tools/curriculum_complete_legal_class_policy_v1.py'):sha(ROOT/'tools/curriculum_complete_legal_class_policy_v1.py'),
        str(ROOT/'tools/curriculum_riichi_class_policy_v1.py'):sha(ROOT/'tools/curriculum_riichi_class_policy_v1.py')}
    def checked(path):
        path=Path(path).resolve();actual=sha(path)
        if bound.get(str(path))!=actual:raise ValueError('Completed comparison input changed')
        files[str(path)]=actual;return read(path)
    done=checked(comparison/'summary-private.json')
    if done['status']!='actual_same16_development_points_and_full_validation_summarized':
        raise ValueError('Actual completed comparison required')
    journals=checked(comparison/'selected-journal-points-private.json');steps=done['actual_steps']
    regs=[Path(p) for p in bound if p.endswith('/fit-plan/registered.json')]
    if len(regs)!=1:raise ValueError('Exactly one actual registration required')
    reg=checked(regs[0]);c=reg['curriculum'];ready=checked(ROOT/c['data_ready'])
    mass=c['loss_mass']
    if not math.isclose(sum(mass.values()),1,abs_tol=1e-10):raise ValueError('Registered normalized source masses required')
    counts=defaultdict(Counter);validation={};source_totals={};source_groups={}
    for split in ('train','validation'):
        spec=ready['files'][split];path=(ROOT/spec['path']).resolve();before=path.stat();digest=hashlib.sha256()
        total=Counter();groups=defaultdict(set);seen=set()
        with path.open('rb') as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():continue
                row=json.loads(line);rid=row['_meta']['id'];q=row['questions']['action'];source=q['src']
                if rid in seen or source not in mass:raise ValueError('Distinct source-aligned corpus required')
                seen.add(rid);total[source]+=1;groups[source].add(row['_meta']['group_id'])
                for op in set(v['type'] for v in q['criteria'].values()) & set(OPS):
                    kind=label_kind(q,op);counts[(split,source,op)][kind]+=1
                    if split=='validation':validation[rid]=q
        after=path.stat()
        if (digest.hexdigest()!=spec['sha256'] or len(seen)!=spec['records']
                or dict(total)!=ready['coverage'][split]['sources']
                or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns)):
            raise ValueError('Actual frozen corpus bytes, source counts or stability differ')
        files[str(path)]=spec['sha256'];source_totals[split]=total;source_groups[split]=groups
    decisions=defaultdict(Counter);original=defaultdict(Counter)
    for step in steps:
        folder=ROOT/c['out']/'training-run/training_eval'/f'step-{step:06d}'
        summary_path=folder/'summary.json';summary=read(summary_path)
        if summary['optimizer_step']!=step or summary['clean']['n']!=ready['files']['validation']['records']:
            raise ValueError('Complete actual saved validation checkpoint required')
        path=folder/'development/rows.json';before=path.stat();raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        predictions=json.loads(raw);del raw;seen=set()
        for p in predictions:
            rid=p['id']
            if rid in seen:raise ValueError('Duplicate cached prediction')
            seen.add(rid)
            if rid not in validation:continue
            q=validation[rid];keys=p['keys'];probabilities=p['p']
            if (len(keys)!=len(set(keys)) or set(keys)!=set(q['criteria']) or len(keys)!=len(probabilities)
                    or any(not math.isfinite(v) or v<0 for v in probabilities)
                    or abs(sum(probabilities)-1)>1e-5):raise ValueError('Actual finite complete cached menu required')
            types=[q['criteria'][k]['type'] for k in keys];chosen,_=select(types,probabilities)
            actual=types[chosen];source=q['src']
            for op in set(types)&set(OPS):
                kind=label_kind(q,op);result=decisions[(step,source,op,kind)]
                result['menus']+=1;result['selected_operation']+=actual==op
                if source.startswith('curriculum2:') and q.get('target') is None:
                    truth=q['criteria'][q['label']]['type']
                    result_key='tp' if truth==op and actual==op else 'fn' if truth==op else 'fp' if actual==op else 'tn'
                    original[(step,op)][result_key]+=1
        if (len(seen)!=ready['files']['validation']['records'] or not set(validation)<=seen
                or (before.st_size,before.st_mtime_ns)!=(path.stat().st_size,path.stat().st_mtime_ns)):
            raise ValueError('Complete stable cached predictions required')
        for op in OPS:
            expected=journals['validation'][str(step)]['original_calls_operation_type_counts'][op]
            if any(original[(step,op)][key]!=expected.get(key,0) for key in ('tp','fn','tn','fp')):
                raise ValueError('Cached decision reconstruction differs from actual journal')
        files[str(path)]=digest;files[str(summary_path)]=sha(summary_path)
    categories=('hard_positive','hard_negative','soft_preferred','soft_other_preferred','soft_tied',
        'soft_operation_unlabelled','soft_no_other_labelled')
    coverage=[];choices=[]
    for (split,source,op),values in sorted(counts.items()):
        n=source_totals[split][source]
        coverage.append(dict(split=split,source=source,operation=op,source_menus=n,
            source_families=len(source_groups[split][source]),legal_operation_menus=sum(values.values()),
            registered_source_mass=mass[source],legal_window_source_mass=mass[source]*sum(values.values())/n,
            hard_positive_source_mass=mass[source]*values['hard_positive']/n,
            **{kind:values[kind] for kind in categories}))
    for (step,source,op,kind),values in sorted(decisions.items()):
        if values['menus']!=counts[('validation',source,op)][kind]:raise ValueError('Stable cached subgroup denominator required')
        choices.append(dict(step=step,source=source,operation=op,label_kind=kind,
            menus=values['menus'],selected_operation=values['selected_operation'],
            selection_rate=values['selected_operation']/values['menus']))
    out.mkdir(parents=True)
    for name,rows in (('kan-label-coverage.csv',coverage),('kan-cached-decisions.csv',choices)):
        with (out/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Kan label coverage and cached decisions','',
        'All counts come from the frozen training/validation corpora and already saved validation probabilities. '
        'No inference or new games were run. Source preference differences are not proof of contradictory labels on identical states.', '',
        '## Training coverage by source','',
        '| Source | Operation | Legal menus | Hard positive | Hard negative | Soft preferred | Soft other preferred | Soft tied | Soft unlabelled | Soft no other |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in coverage:
        if row['split']=='train':lines.append('| '+row['source']+' | '+row['operation']+' | '+str(row['legal_operation_menus'])+
            ' | '+' | '.join(str(row[k]) for k in categories)+' |')
    lines+=['','## Validation selection by label category','',
        '| Operation | Label category | Menus | '+' | '.join('Selected '+str(step) for step in steps)+' |',
        '| --- | --- | ---: | '+' | '.join('---:' for _ in steps)+' |']
    for op in OPS:
        for kind in categories:
            points=[]
            for step in steps:
                selected=[r for r in choices if r['step']==step and r['operation']==op and r['label_kind']==kind]
                n=sum(r['menus'] for r in selected);yes=sum(r['selected_operation'] for r in selected)
                points.append((n,yes))
            if points[0][0]:lines.append('| '+op+' | '+kind+' | '+str(points[0][0])+' | '+
                ' | '.join(f'{yes}/{n} ({yes/n:.2%})' for n,yes in points)+' |')
    lines+=['','Hard positives/negatives follow cached action-type labels. Soft preferred/other/tied compares the largest '
        'labelled candidate probability of that operation with the largest labelled alternative. Legal but unlabelled operations '
        'are separate; they are not negative labels. None of these categories proves real kan profitability. '
        'The CSV also reports source masses: these describe sampling/loss allocation, not realized reward.', '',
        'Active training, its 2500-update budget, dataset and final confirmation remain unchanged. Raw hands and probabilities stay private.','']
    (out/'kan-label-balance.md').write_text('\n'.join(lines))
    write(out/'summary-private.json',dict(status='frozen_kan_label_coverage_and_actual_cached_choices_audited',
        actual_steps=steps,train_records=ready['files']['train']['records'],validation_records=ready['files']['validation']['records'],
        cached_legacy_operation_confusions_match_actual_journals=True,unlabelled_soft_operations_not_counted_negative=True,
        same_state_label_contradictions_proved=False,kan_profitability_proved=False,new_games=0,model_loaded=False,
        student_updates=0,recipe_unchanged=True,allocation=allocation))
    for path in out.iterdir():files[str(path)]=sha(path)
    write(out/'completion-binding-private.json',dict(files=files))
    print(json.dumps(dict(status='frozen_kan_label_coverage_and_actual_cached_choices_audited',
        steps=steps,coverage=coverage,cached_choices=choices)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();audit(args.comparison,args.out)
