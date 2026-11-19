"""TRAIN-only paired-world leave-one-out audit of existing ordinary-fourth labels."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def validate_config(config):
    expected={'schema':'endgame-training-label-stability-v1','training_positions':55,'strict_positions':19,
        'legacy_positions':23,'legacy_worlds':4,'native_positions':32,'native_worlds':16,
        'terminal_utility':[1.1,1.,0.,-.05],'method':'paired_world_leave_one_out',
        'tie_break':'original_ordered_menu','optimizer_updates':0,'new_model_queries':0,'new_simulations':0}
    if any(config.get(k)!=v for k,v in expected.items()) or any(config.get(k) is not False for k in (
            'validation_data_used','fit_registered','independent_confirmation','raw_publication')):
        raise ValueError('Only the registered complete TRAIN55 zero-query label audit is enabled')


def outcome_arrays(label,keys,seat,worlds,parent_identity):
    from jevmahjong.endgame.context import placements,placement_objective,terminal_utility
    if (label['status']!='complete_terminal_menu' or label['source']['split']!='train'
            or label['source']['seat']!=seat or type(seat) is not int or not 0<=seat<4
            or label['baseline_identity_sha256']!=parent_identity
            or label['worlds_per_action']!=worlds or not label['continued_until_native_match_done']
            or list(label['record']['questions']['action']['criteria'])!=keys
            or label['actual_actions']!=len(keys) or len(label['outcomes'])!=len(keys)):
        raise ValueError('Raw complete TRAIN menu identity or paired-world contract differs')
    utilities={};escape={}
    for key,outcomes in zip(keys,label['outcomes']):
        if len(outcomes)!=worlds:raise ValueError('Incomplete paired worlds')
        utilities[key]=[];escape[key]=[]
        for value in outcomes:
            if (value['ledger_passed'] is not True or placements(value['scores'])!=value['ranks']
                    or value['utility']!=terminal_utility(value['ranks'][seat])
                    or value['hand_history']['final_scores']!=value['scores']):
                raise ValueError('Raw terminal scores, ranks, ledger or history differ')
            utilities[key].append(placement_objective(value['ranks'][seat]))
            escape[key].append(float(value['ranks'][seat]<4))
    return utilities,escape


def target_for_worlds(reference,utilities,escape,eligible,indices):
    from jevmahjong.endgame.reference import escape_supported_keys,greedy_improvement_target
    if (not indices or list(utilities)!=list(reference) or list(escape)!=list(reference)
            or any(len(utilities[k])!=len(escape[k]) for k in reference)
            or len({len(v) for v in utilities.values()})!=1
            or any(not math.isfinite(v) for values in utilities.values() for v in values)
            or any(not math.isfinite(v) or not 0<=v<=1 for values in escape.values() for v in values)):
        raise ValueError('Paired finite-world arrays differ or are nonfinite')
    n=len(next(iter(utilities.values())))
    if len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<n for i in indices):
        raise ValueError('Invalid common paired-world indices')
    mean=lambda values:sum(values[i] for i in indices)/len(indices)
    q={k:mean(v) for k,v in utilities.items()};e={k:mean(v) for k,v in escape.items()}
    baseline=max(eligible,key=reference.get);safe=escape_supported_keys(reference,e,eligible)
    target=greedy_improvement_target(reference,q,safe)
    return {'utilities':q,'escape':e,'target':target,'choice':max(target,key=target.get),
        'baseline':baseline,'support':[k for k in target if target[k]>0],
        'strict':max(q[k] for k in safe)>q[baseline]+1e-12}


def leave_one_out(reference,utilities,escape,eligible):
    n=len(next(iter(utilities.values())))
    if n<2:raise ValueError('At least two paired worlds are required')
    full=target_for_worlds(reference,utilities,escape,eligible,list(range(n)));worlds=[]
    for heldout in range(n):
        current=target_for_worlds(reference,utilities,escape,eligible,[i for i in range(n) if i!=heldout])
        chosen,baseline=current['choice'],full['baseline']
        worlds.append({'heldout_index':heldout,'choice':chosen,'strict':current['strict'],
            'support_matches_full':current['support']==full['support'],
            'choice_matches_full':chosen==full['choice'],
            'heldout_utility_difference':utilities[chosen][heldout]-utilities[baseline][heldout],
            'heldout_escape_difference':escape[chosen][heldout]-escape[baseline][heldout]})
    mean=lambda field:sum(v[field] for v in worlds)/n
    return {'full':full,'worlds':worlds,'world_count':n,
        'strict_remaining_fraction':mean('strict'),'support_matches_full_fraction':mean('support_matches_full'),
        'choice_matches_full_fraction':mean('choice_matches_full'),
        'heldout_mean_utility_difference':mean('heldout_utility_difference'),
        'heldout_mean_escape_difference':mean('heldout_escape_difference'),
        'full_choice_mean_utility_difference':full['utilities'][full['choice']]-full['utilities'][full['baseline']]}


def aggregate(cases):
    if not cases:raise ValueError('Do not report an empty registered cohort')
    fields=('strict_remaining_fraction','support_matches_full_fraction','choice_matches_full_fraction',
        'heldout_mean_utility_difference','heldout_mean_escape_difference','full_choice_mean_utility_difference')
    means={field:sum(c['audit'][field] for c in cases)/len(cases) for field in fields}
    return dict(means,positions=len(cases),paired_worlds=sum(c['audit']['world_count'] for c in cases),
        full_strict_positions=sum(c['audit']['full']['strict'] for c in cases),
        strict_retained_in_every_loo=sum(c['audit']['strict_remaining_fraction']==1. for c in cases),
        choice_stable_in_every_loo=sum(c['audit']['choice_matches_full_fraction']==1. for c in cases),
        support_stable_in_every_loo=sum(c['audit']['support_matches_full_fraction']==1. for c in cases),
        positions_positive_heldout_utility=sum(c['audit']['heldout_mean_utility_difference']>1e-12 for c in cases),
        positions_negative_heldout_utility=sum(c['audit']['heldout_mean_utility_difference']< -1e-12 for c in cases),
        aggregation='equal_position_weight_not_world_count_weight')


def audit(config):
    validate_config(config)
    from jevmahjong.endgame.prepare import read_jsonl,checksum
    from jevmahjong.endgame.collect import write,digest
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.public_policy import action_keys
    from jevmahjong.endgame.native_terminal import is_native_terminal
    from desktop.jev_sidecar.unified_policy import identity
    out,run=Path(config['out']),Path(config['source_data_run'])
    if any((out/name).exists() for name in ('protocol-private.json','summary-private.json')):
        raise ValueError('Inspect an existing audit rather than silently restart')
    preparation=json.loads((run/'preparation-private.json').read_text())
    if checksum(run/'train.jsonl')!=preparation['files']['train']['sha256']:
        raise ValueError('Prepared training records changed')
    source_config=json.loads((run/'config-private.json').read_text())
    parent=bundle_identity(source_config['checkpoint']);released=identity()
    if parent['identity_sha256'] not in (released['identity_sha256'],released['source_identity_sha256']):
        raise ValueError('Continuation labels must use formal v8')
    reference=json.loads((run/'reference-targets-private.json').read_text())
    if reference['identity_sha256']!=parent['identity_sha256']:raise ValueError('Original reference model changed')
    records=[r for r in read_jsonl(run/'train.jsonl') if r['questions']['action']['src'].startswith('endgame_terminal_fourth')]
    if len(records)!=55 or len({r['_meta']['id'] for r in records})!=55:
        raise ValueError('The complete original55 ordinary-fourth training set changed')
    source_paths=set(ROOT.glob('jevmahjong/**/*.py'))|set(ROOT.glob('desktop/jev_sidecar/**/*.py'))
    source_paths.update(ROOT/name for name in ('tools/audit_endgame_training_label_stability.py',
        'configs/training/endgame-training-label-stability-v1.json','tests/core/test_endgame_training_label_stability.py'))
    source_paths.update(run/name for name in ('train.jsonl','config-private.json','preparation-private.json','reference-targets-private.json'))
    raw=[];counts=Counter()
    for row in records:
        meta=row['_meta'];native=is_native_terminal(meta);cohort='native' if native else 'legacy'
        if meta['split']!='train' or meta['source_initial_placement']!=4:raise ValueError('Nontraining or nonfourth record')
        expected_worlds=16 if native else 4
        provenance=meta['terminal_label_provenance']
        if provenance['worlds_per_action']!=expected_worlds:raise ValueError('Registered world count changed')
        label_root=Path(source_config['native_terminal_label_root']) if native else run
        path=label_root/'labels'/(meta['id'].split(':',1)[1]+'.json')
        if checksum(path)!=provenance['file_sha256']:raise ValueError('Raw source label changed after admission')
        source_paths.add(path);raw.append((row,json.loads(path.read_text()),cohort,expected_worlds));counts[cohort]+=1
    if counts!={'legacy':23,'native':32}:raise ValueError('Registered complete source cohorts differ')
    protocol={'configuration':config,'parent_identity':parent,'training_ids':[r['_meta']['id'] for r in records],
        'source_sha256':{str(p.resolve()):checksum(p) for p in sorted(source_paths)},
        'validation_data_used':False,'optimizer_updates':0,'new_simulations':0,'new_model_queries':0}
    write(out/'protocol-private.json',protocol)
    cases=[];strict_counts=Counter()
    for row,label,cohort,world_count in raw:
        meta,q=row['_meta'],row['questions']['action'];rid=meta['id'];keys=list(q['criteria']);seat=label['source']['seat']
        request_field='student_request' if cohort=='native' else 'request'
        if digest(label['source'][request_field])!=label['source_request_sha256']:
            raise ValueError('Raw source public request fingerprint changed')
        prior=reference['rows'][rid]['probabilities']
        utilities,escape=outcome_arrays(label,keys,seat,world_count,parent['identity_sha256'])
        result=leave_one_out(prior,utilities,escape,action_keys(row));full=result['full']
        if (any(abs(full['utilities'][k]-meta['expected_terminal_utilities'][k])>1e-12 for k in keys)
                or any(abs(full['escape'][k]-meta['terminal_escape4_probabilities'][k])>1e-12 for k in keys)
                or any(abs(full['target'][k]-q['target'][k])>1e-12 for k in keys)
                or full['choice']!=q['label'] or full['strict'] is not meta['strict_original_policy_improvement']
                or (q['src']=='endgame_terminal_fourth_improve')!=full['strict']):
            raise ValueError('Full-world source utility/escape/target/strict partition does not reproduce training')
        strict_counts[cohort]+=full['strict']
        cases.append({'id':rid,'cohort':cohort,'strict':full['strict'],'audit':result})
    if strict_counts!={'legacy':9,'native':10}:raise ValueError('Original9/native10 strict definition changed')
    groups={'all':cases,'legacy':[c for c in cases if c['cohort']=='legacy'],
        'native':[c for c in cases if c['cohort']=='native'],'strict':[c for c in cases if c['strict']],
        'keep':[c for c in cases if not c['strict']]}
    groups.update({cohort+'_'+kind:[c for c in cases if c['cohort']==cohort and c['strict']==flag]
        for cohort in ('legacy','native') for kind,flag in (('strict',True),('keep',False))})
    for path,expected in protocol['source_sha256'].items():
        if checksum(path)!=expected:raise ValueError('Read-only audit dependency changed')
    if bundle_identity(source_config['checkpoint'])!=parent:raise ValueError('Formal model files changed')
    write(out/'cases-private.json',cases)
    summary={'status':'complete_training_label_stability_audit','training_positions':55,'strict_positions':19,
        'cohorts':{key:aggregate(values) for key,values in groups.items()},
        'full_world_targets_and_ledgers_reproduced':True,'source_and_parent_fingerprints_unchanged':True,
        'validation_data_used':False,'optimizer_updates':0,'new_model_queries':0,'new_simulations':0,
        'independent_confirmation':False,'deployed':False,'raw_publication':False,
        'uncertainty':'descriptive conditional paired-world LOO, not independent strength or unbiased population effect'}
    write(out/'summary-private.json',summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Label stability processing requires a scheduled GPU allocation')
    from tools.prepare_native_context_replay import allocation_metadata
    description=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True)
    print(json.dumps({'node':socket.gethostname(),'allocation':allocation_metadata(description,socket.gethostname())}),flush=True)
    print(json.dumps(audit(json.loads(args.config.read_text())),indent=2),flush=True)


if __name__=='__main__':main()
