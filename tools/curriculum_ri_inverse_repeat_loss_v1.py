"""Per-original replay weights and actual optimizer loss accounting.

The nineteen unweighted objectives remain unchanged. Weight annotations belong
only to materialized questions; raw requests and model text are never edited.
This module does not select a parent, register a fit or update model parameters.
"""
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_progression_parent_loss_v1 as retained

METHOD='per_original_inverse_repeat_weights_exact_retained19_objectives_v1'
FIELD='_RI_inverse_repeat_weight_v1'
IDENTIFIER='_RI_inverse_repeat_original_id_v1'
MARKER='_RI_inverse_repeat_method_v1'


def attach(encoded,weight,identifier):
    if (type(weight) not in (int,float) or not math.isfinite(weight) or not 0<weight<=16
            or not isinstance(identifier,str) or not identifier):
        raise ValueError('A positive bounded real weight and original record ID required')
    for q in encoded['questions']:
        if (q.get(retained.MARKER)!=retained.METHOD or q['src'] not in retained.MASS
                or any(k in q for k in (FIELD,IDENTIFIER,MARKER))):
            raise ValueError('Pristine retained19 loss annotation required')
        q.update({FIELD:float(weight),IDENTIFIER:identifier,MARKER:METHOD})
    return encoded


def weighted_loss(logits,question,device,observer=None):
    import torch
    if (question.get(MARKER)!=METHOD or FIELD not in question or IDENTIFIER not in question
            or type(question[FIELD]) is not float or not math.isfinite(question[FIELD])
            or not 0<question[FIELD]<=16 or not question[IDENTIFIER]):
        raise ValueError('Explicit valid inverse-repeat weight annotation required')
    weight=question[FIELD];components=retained.parts(logits,question,device)
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=question['src'],record_id=question[IDENTIFIER],example_weight=weight,
            **{key:float(value.detach()) for key,value in components.items()}))
    return components['objective']*weight


def summarize(pending):
    if not pending:raise ValueError('Actual optimized examples required')
    grouped=defaultdict(list)
    for row in pending:
        if (row['source'] not in retained.MASS or not row['record_id']
                or not math.isfinite(row['example_weight']) or not 0<row['example_weight']<=16
                or any(not math.isfinite(row[k]) for k in retained.COMPONENTS)
                or abs(row['objective']-row['standard_target_ce']-row['signed_objective_minus_standard_ce'])>1e-5):
            raise ValueError('Valid actual weighted loss components required')
        grouped[row['source']].append(row)
    n=len(pending)
    weighted=lambda key:sum(r['example_weight']*r[key] for r in pending)/n
    return dict(loss=weighted('objective'),ce=weighted('standard_target_ce'),
        signed_objective_adjustment=weighted('signed_objective_minus_standard_ce'),
        source_counts={s:len(v) for s,v in grouped.items()},
        source_loss_components={s:{k:sum(r[k] for r in v)/len(v) for k in retained.COMPONENTS} for s,v in grouped.items()},
        source_weighted_component_contributions={s:{k:sum(r['example_weight']*r[k] for r in v)/n
            for k in retained.COMPONENTS} for s,v in grouped.items()},
        effective_source_mass={s:sum(r['example_weight'] for r in v)/n for s,v in grouped.items()},
        source_example_weight_range={s:[min(r['example_weight'] for r in v),max(r['example_weight'] for r in v)] for s,v in grouped.items()},
        loss_method=METHOD,unweighted_objective_method=retained.METHOD)


def preflight():
    from tools.capture_curriculum_paired_terminal_v2 import resource
    from tools.curriculum_transfer import read,sha
    from tools.curriculum_calls_post_reach_data_v1 import write
    allocation=resource();parent=ROOT/'runs/curriculum-progression-parent-retention-v1'
    registration=parent/'fit-plan/registered.json';reg=read(registration)
    planroot=ROOT/'runs/curriculum-ri-every-update-replay-v1'
    binding_path=planroot/'completion-binding-private.json';binding=read(binding_path)
    if any(sha(p)!=h for p,h in binding['files'].items()):raise ValueError('Frozen exposure plan changed')
    sys.path[:0]=[reg['source'],reg['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    from jevany.data import materialize
    from jevany.model import load_tokenizer,encode
    receipt=read(reg['training_config']['data_receipt']);path=Path(receipt['files']['train']['path'])
    if sha(path)!=receipt['files']['train']['sha256']:raise ValueError('Original TRAIN changed')
    out=ROOT/'runs/curriculum-ri-inverse-repeat-loss-v1'
    if out.exists():raise ValueError('Never replace a preflight attempt')
    out.mkdir()
    with path.open() as stream:rows=[json.loads(line) for line in stream]
    if len(rows)!=87595:raise ValueError('All original TRAIN rows required')
    plans=[read(planroot/f'epoch-{epoch:02d}-private.json') for epoch in (1,2)]
    tokenizer=load_tokenizer(parent/'stage3-fit/training-run/checkpoints/step-001200')
    cached=[];representatives={};source_weight_sets=defaultdict(set)
    for plan in plans:
        if set(plan['weights_by_original_index'])!={str(i) for i in range(len(rows))}:raise ValueError('Every original weight required')
        for i,row in enumerate(rows):source_weight_sets[row['questions']['action']['src']].add(plan['weights_by_original_index'][str(i)])
    for i,row in enumerate(rows):
        rec=retained.annotate(materialize(row),row);q=rec['questions'][0]
        z=torch.linspace(-.9,1.1,len(q['keys']),dtype=torch.float64)
        components={k:float(v) for k,v in retained.parts(z,q,'cpu').items()}
        cached.append((rec,z,components));representatives.setdefault(q['src'],i)
        if (i+1)%20000==0:print(json.dumps({'phase':'actual_unique_TRAIN_objectives','rows':i+1}),flush=True)
    checked_token_inputs=0;checked_gradients=0
    for source,i in representatives.items():
        rec,z,components=cached[i];raw=rows[i]
        base_encoding=encode(tokenizer,rec,max_state=4096,max_branch=8192,strict=True)
        for weight in source_weight_sets[source]:
            work=attach(copy.deepcopy(rec),weight,raw['_meta']['id'])
            if encode(tokenizer,work,max_state=4096,max_branch=8192,strict=True)!=base_encoding:
                raise ValueError('Loss-only annotations changed model token encoding')
            a=z.detach().clone().requires_grad_(True);b=z.detach().clone().requires_grad_(True)
            actual=weighted_loss(a,work['questions'][0],'cpu')
            expected=retained.parts(b,rec['questions'][0],'cpu')['objective']*weight
            if not torch.equal(actual,expected):raise ValueError('Actual weighted objective changed')
            ga,=torch.autograd.grad(actual,a);gb,=torch.autograd.grad(expected,b)
            if not torch.equal(ga,gb):raise ValueError('Actual weighted logit gradient changed')
            checked_token_inputs+=1;checked_gradients+=1
    largest_objective_error=0.;largest_journal_error=0.;exposures=0;updates=0
    for epoch,plan in enumerate(plans,1):
        pending=[];actual_losses=[];expected_ce=[]
        for position,i in enumerate(plan['indices'],1):
            rec,z,components=cached[i];weight=plan['weights_by_original_index'][str(i)]
            work=attach({'questions':[dict(rec['questions'][0])]},weight,rows[i]['_meta']['id'])
            value=weighted_loss(z.detach().clone().requires_grad_(True),work['questions'][0],'cpu',pending)
            actual=float(value.detach());actual_losses.append(actual);expected_ce.append(weight*components['standard_target_ce'])
            largest_objective_error=max(largest_objective_error,abs(actual-weight*components['objective']))
            if len(pending)==73:
                observation=summarize(pending)
                largest_journal_error=max(largest_journal_error,
                    abs(observation['loss']-sum(actual_losses)/73),abs(observation['ce']-sum(expected_ce)/73))
                if abs(observation['loss']-observation['ce']-observation['signed_objective_adjustment'])>1e-10:
                    raise ValueError('Actual weighted signed-loss accounting differs')
                pending.clear();actual_losses.clear();expected_ce.clear();updates+=1
            exposures+=1
            if position%30000==0:print(json.dumps({'phase':'actual_weighted_loss_and_journal','epoch':epoch,'exposures':position}),flush=True)
        if pending:raise ValueError('Unfinished optimizer observation')
    if exposures!=180456 or updates!=2472 or largest_objective_error>1e-12 or largest_journal_error>1e-12:
        raise ValueError('Incomplete or incorrect actual weighted-loss/journal pass')
    report={'status':'complete_actual_TRAIN_inverse_repeat_loss_and_journal_preflight_not_model_training',
        'unique_TRAIN_rows':len(rows),'actual_exposure_losses_checked':exposures,'optimizer_observations_checked':updates,
        'sources':len(representatives),'distinct_weight_source_gradients_checked':checked_gradients,
        'model_token_encodings_unchanged':checked_token_inputs,'maximum_weighted_objective_error':largest_objective_error,
        'maximum_actual_journal_error':largest_journal_error,'synthetic_logits_not_training_or_native_curves':True,
        'model_loaded':False,'student_updates':0,'new_games':0,'allocation':allocation,
        'live_trial_changed':False,'actual_trainer_epoch_mapping_and_fit_not_yet_integrated':True}
    write(out/'summary-private.json',report)
    files=dict(binding['files']);files[str(binding_path)]=sha(binding_path)
    for p in (Path(__file__).resolve(),ROOT/'tests/core/test_curriculum_ri_inverse_repeat_loss_v1.py',
              registration,out/'summary-private.json'):files[str(p)]=sha(p)
    write(out/'completion-binding-private.json',{'files':files})
    print(json.dumps({k:v for k,v in report.items() if k!='allocation'}),flush=True)


if __name__=='__main__':preflight()
