"""Replace calls13 source scalars with frozen, capped label-class scalars.

Attach annotations only after the original encoder returns. Keep all existing
thirteen-source objective components. This module admits no parent or fit.
"""
from collections import Counter, defaultdict
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_call_type_balance_v1 as recipe
from tools import curriculum_calls_classroute_v1 as original

METHOD='calls13_existing_objectives_replacement_call_type_scalars_v1'
FIELD='_call_type_weight_v1'
IDENTIFIER='_call_type_original_id_v1'
CLASS='_call_type_observed_class_v1'
MARKER='_call_type_scalar_method_v1'


class CallTypeIndex:
    def __init__(self,rows,entries):
        expected,self.recipe=recipe.build(rows)
        if entries!=expected:
            raise ValueError('All original rows and exact frozen source/class scalars required')
        self.entries={row['_meta']['id']:entry for row,entry in zip(rows,entries)}

    def attach(self,encoded,request):
        identifier=request['_meta']['id'];entry=self.entries[identifier]
        if len(encoded['questions'])!=1:
            raise ValueError('One original hard action question required')
        q=encoded['questions'][0];raw=request['questions'][q['qid']]
        source=original.key(raw['src'])
        kind=recipe.category(source,raw['criteria'][raw['label']]['type']) if source in recipe.SOURCES else 'unchanged_source'
        if (q.get(original.MARKER)!=original.METHOD or q['src']!=raw['src']
                or set(q['keys'])!=set(raw['criteria']) or len(set(q['keys']))!=len(q['keys'])
                or q['keys'][q['label']]!=raw['label'] or q.get('target') is not None
                or entry['source']!=source or entry['label_class']!=kind
                or any(key in q for key in (FIELD,IDENTIFIER,CLASS,MARKER))):
            raise ValueError('Pristine aligned calls13 encoded target required')
        q.update({FIELD:float(entry['weight']),IDENTIFIER:identifier,CLASS:kind,MARKER:METHOD})
        return encoded

    def encode_batch(self,encoder,model,tok,args,chunk,epoch):
        if (args.batch!=1 or getattr(args,'rlcr',False)
                or any(getattr(args,key)!=0 for key in ('p_none','p_none_distract','p_distract','p_none_pair'))):
            raise ValueError('Single-record supervised encoding without augmentation required')
        variants=encoder(model,tok,args,chunk,epoch)
        if len(variants)!=len(chunk):
            raise ValueError('Original encoder changed record correspondence')
        for variant,raw in zip(variants,chunk):
            self.attach(variant.rec,raw)
        return variants


def weighted_loss(logits,question,device,observer=None):
    import torch
    if (question.get(MARKER)!=METHOD or type(question.get(FIELD)) is not float
            or not math.isfinite(question[FIELD]) or not 0<question[FIELD]<=16
            or not question.get(IDENTIFIER) or not question.get(CLASS)):
        raise ValueError('Frozen bounded per-original replacement scalar required')
    components=original.parts(logits,question,device)
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=original.key(question['src']),record_id=question[IDENTIFIER],
            label_class=question[CLASS],example_weight=question[FIELD],
            **{key:float(value.detach()) for key,value in components.items()}))
    # Replacement: do not multiply the old uniform source weight again.
    return components['objective']*question[FIELD]


def summarize(pending):
    if not pending:raise ValueError('Actual optimized records required')
    groups=defaultdict(list);classes=defaultdict(Counter)
    for row in pending:
        if (row['source'] not in original.MASS or not row['record_id'] or not row['label_class']
                or not math.isfinite(row['example_weight']) or not 0<row['example_weight']<=16
                or any(not math.isfinite(row[k]) for k in original.COMPONENTS)):
            raise ValueError('Finite complete weighted optimizer components required')
        groups[row['source']].append(row);classes[row['source']][row['label_class']]+=1
    n=len(pending)
    weighted=lambda key:sum(row['example_weight']*row[key] for row in pending)/n
    return dict(loss=weighted('objective'),ce=weighted('standard_action_ce'),
        source_counts={s:len(v) for s,v in groups.items()},
        source_class_counts={s:dict(v) for s,v in classes.items()},
        effective_source_mass={s:sum(r['example_weight'] for r in v)/n for s,v in groups.items()},
        source_weighted_component_contributions={s:{key:sum(r['example_weight']*r[key] for r in v)/n
            for key in original.COMPONENTS} for s,v in groups.items()},
        loss_method=METHOD,unweighted_objective_method=original.METHOD)


def preflight():
    import copy
    import json
    import subprocess
    import tomllib
    from tools.capture_curriculum_paired_terminal_v2 import resource
    from tools.curriculum_transfer import read,sha
    from tools.curriculum_calls_post_reach_data_v1 import write
    allocation=resource();prior=ROOT/'runs/curriculum-call-type-balance-v1/preflight-v1'
    binding=read(prior/'completion-binding-private.json')
    if any(sha(p)!=h for p,h in binding['files'].items()):raise ValueError('Frozen class index changed')
    registration=ROOT/'runs/curriculum-calls-classroute-v1/fit-plan/registered.json';reg=read(registration)
    dependency_files=[ROOT/'tools'/name for name in (
        'curriculum_calls_classroute_v1.py','curriculum_calls_riichi_retention_v1.py',
        'curriculum_riichi_class_loss.py','curriculum_calls_post_reach_data_v1.py')]
    dependency_hashes={str(p):sha(p) for p in dependency_files}
    for p,h in dependency_hashes.items():
        if p in reg['inputs'] and reg['inputs'][p]!=h:
            raise ValueError('Registered original loss dependency changed: '+p)
    args=tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    rows=[json.loads(line) for line in Path(args['data']).open()]
    entries=read(prior/'row-index-and-weights-private.json');index=CallTypeIndex(rows,entries)
    sys.path[:0]=[reg['source'],reg['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    from jevany.data import materialize
    from jevany.model import encode,load_tokenizer
    out=ROOT/'runs/curriculum-call-type-loss-v1/preflight-v1'
    out.mkdir(parents=True,exist_ok=False)
    representatives={};unaffected=0;checked=0;maximum_error=0.;sums=defaultdict(float)
    for row,entry in zip(rows,entries):
        plain=original.annotate(materialize(row),row)
        encoded=index.attach(copy.deepcopy(plain),row);question=encoded['questions'][0]
        logits=torch.linspace(-.9,1.1,len(question['keys']),dtype=torch.float64)
        expected=original.parts(logits,plain['questions'][0],'cpu')['objective']*entry['weight']
        actual=weighted_loss(logits,question,'cpu')
        maximum_error=max(maximum_error,abs(float(actual-expected)))
        source=entry['source'];sums[source]+=entry['weight']/len(rows)
        if source not in recipe.SOURCES:
            old=original.weighted_loss(logits,plain['questions'][0],'cpu',
                {s:d['previous_uniform_row_weight'] for s,d in index.recipe['sources'].items()})
            if not torch.equal(actual,old):raise ValueError('An unaffected source objective changed')
            unaffected+=1
        representatives.setdefault((source,entry['label_class']),(row,plain,encoded,logits,entry))
        checked+=1
        if checked%15000==0:print(json.dumps(dict(stage='actual_TRAIN_replacement_objectives',rows=checked)),flush=True)
    if maximum_error!=0 or any(abs(sums[s]-original.MASS[s])>1e-12 for s in original.MASS):
        raise ValueError('Exact intended replacement loss and source mass required')
    tokenizer=load_tokenizer(ROOT/'runs/curriculum-calls-classroute-v1/stage2-fit/training-run/checkpoints/step-000388')
    gradients=0;token_checks=0
    for row,plain,encoded,logits,entry in representatives.values():
        options=dict(max_state=args['max_state'],max_branch=args['max_branch'],strict=True)
        if encode(tokenizer,plain,**options)!=encode(tokenizer,encoded,**options):
            raise ValueError('Loss-only annotations changed actual token encoding')
        a=logits.clone().requires_grad_();b=logits.clone().requires_grad_()
        actual=weighted_loss(a,encoded['questions'][0],'cpu')
        expected=original.parts(b,plain['questions'][0],'cpu')['objective']*entry['weight']
        if not torch.equal(torch.autograd.grad(actual,a)[0],torch.autograd.grad(expected,b)[0]):
            raise ValueError('Replacement logit gradient differs')
        gradients+=1;token_checks+=1
    tests=subprocess.run([sys.executable,'-m','unittest','tests.core.test_curriculum_call_type_loss_v1'],
        capture_output=True,text=True,timeout=90)
    if tests.returncode:raise RuntimeError(tests.stdout+tests.stderr)
    summary=dict(status='all49022_replacement_objectives_and_representative_gradient_tokens_passed_not_fit',
        method=METHOD,actual_TRAIN_rows=checked,unaffected_source_rows_exact=unaffected,
        representative_source_classes=gradients,representative_token_checks=token_checks,
        maximum_objective_difference=maximum_error,source_mass_unchanged=True,
        validation_and_test_read=False,model_loaded=False,student_updates=0,new_games=0,
        actual_full_model_token_admission_and_optimizer_integration_pending=True,
        live_trial_changed=False,allocation=allocation,test_output=tests.stdout+tests.stderr)
    write(out/'summary-private.json',summary)
    files=dict(binding['files']);files.update(dependency_hashes)
    files[str(prior/'completion-binding-private.json')]=sha(prior/'completion-binding-private.json')
    for p in (Path(__file__).resolve(),ROOT/'tests/core/test_curriculum_call_type_loss_v1.py',
        Path(materialize.__code__.co_filename),Path(encode.__code__.co_filename),out/'summary-private.json'):
        files[str(p.resolve())]=sha(p)
    if any(sha(p)!=h for p,h in files.items()):raise ValueError('Frozen loss/index prerequisite changed')
    write(out/'completion-binding-private.json',dict(files=files))
    print(json.dumps({k:v for k,v in summary.items() if k!='allocation'}),flush=True)


if __name__=='__main__':preflight()
