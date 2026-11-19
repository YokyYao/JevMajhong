"""Read-only finite-label soft progress in the completed constrained-head pilot."""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def action_expectation(probabilities,values,eligible):
    from jevmahjong.endgame.reference import distribution,restrict_distribution
    distribution(probabilities,list(values))
    if not all(math.isfinite(v) for v in values.values()):raise ValueError('Nonfinite finite-rollout label')
    p=restrict_distribution(probabilities,eligible);chosen=max(eligible,key=p.get)
    return {'greedy_key':chosen,'greedy_value':values[chosen],
            'soft_value':sum(p[k]*values[k] for k in p)}


def summarize(records,predictions,baseline):
    from jevmahjong.endgame.public_policy import action_keys
    from jevmahjong.endgame.native_terminal import is_native_terminal
    if set(predictions)!={r['_meta']['id'] for r in records} or set(baseline)!=set(predictions):
        raise ValueError('Fixed complete validation identities changed')
    strata=defaultdict(list);escapes=defaultdict(list);third=[];changes=0;terminal_changes=0
    for row in records:
        meta,q=row['_meta'],row['questions']['action'];rid=meta['id'];eligible=action_keys(row)
        changed=max(eligible,key=predictions[rid].get)!=max(eligible,key=baseline[rid].get)
        changes+=changed
        if q['src']!='endgame_terminal':continue
        terminal_changes+=changed
        value=action_expectation(predictions[rid],meta['expected_terminal_utilities'],eligible)
        strata[meta['bucket']].append(value['soft_value'])
        if meta['source_initial_placement']==3:third.append(value['soft_value'])
        if meta['source_initial_placement']==4:
            key='native_fourth' if is_native_terminal(meta) else 'existing'
            escapes[key].append(action_expectation(predictions[rid],meta['terminal_escape4_probabilities'],eligible)['soft_value'])
    if (len(records)!=186 or set(strata)!={'win','third','fourth','hold'} or len(third)!=23
            or {k:len(v) for k,v in escapes.items()}!={'existing':43,'native_fourth':8}):
        raise ValueError('Original complete soft-progress cohorts changed')
    mean=lambda v:sum(v)/len(v)
    means={k:mean(v) for k,v in strata.items()}
    return {'records':186,'greedy_choices_changed_from_step0':changes,
        'terminal_greedy_choices_changed_from_step0':terminal_changes,
        'soft_utility_by_stratum':means,'soft_equal_stratum_utility':mean(list(means.values())),
        'soft_third23':mean(third),'soft_fourth_escape':{k:mean(v) for k,v in escapes.items()},
        'finite_labels_not_independent_strength':True}


def audit(out):
    from tools.train_endgame_constrained_head import verify_sources
    from jevmahjong.endgame.prepare import read_jsonl,checksum
    from jevmahjong.endgame.collect import write
    run=ROOT/'runs/endgame-placement-v15';source=ROOT/'runs/endgame-placement-v14'
    protocol=json.loads((run/'protocol-private.json').read_text());verify_sources(protocol)
    final=json.loads((run/'postfit-summary-private.json').read_text())
    if (final['last_step']!=2 or final['selected_step']!=0
            or not final['normal_service_controls']['2']['passed']
            or final['normal_service_controls']['2']['max_absolute_probability_difference']!=0.):
        raise ValueError('Expected completed normal-service-audited unqualified pilot2')
    if (out/'summary-private.json').exists():raise ValueError('Do not rerun completed diagnostic')
    records=read_jsonl(source/'validation.jsonl')
    paths=[run/'training-run/predictions'/f'step-{step:06d}-private.json' for step in (0,1,2)]
    hashes={str(p):checksum(p) for p in [*paths,Path(__file__).resolve()]}
    write(out/'protocol-private.json',{'steps':[0,1,2],'prediction_sha256':hashes,
        'optimizer_updates':0,'new_model_queries':0,'new_simulations':0,'independent_confirmation':False})
    predictions=[json.loads(p.read_text()) for p in paths]
    result={str(step):summarize(records,p,predictions[0]) for step,p in enumerate(predictions)}
    verify_sources(protocol)
    if any(checksum(p)!=expected for p,expected in hashes.items()):raise ValueError('Read-only input changed')
    summary={'status':'soft_progress_readonly_audit_complete','steps':result,'optimizer_updates':0,
        'new_model_queries':0,'new_simulations':0,'independent_confirmation':False,'deployed':False,'raw_publication':False}
    write(out/'summary-private.json',summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Soft progress audit requires a scheduled GPU allocation')
    print(json.dumps(audit(args.out),indent=2),flush=True)


if __name__=='__main__':main()
