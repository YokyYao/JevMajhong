"""Independently recount native events before accepting a new-adapter result."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
from jevmahjong.core.io import digest,write
from jevmahjong.training.provenance import verify
from .unified_contract import checked_candidate,assert_same_candidate,verify_files,bundle,read


def recount(events,seat,gamma):
    """Rebuild scores, riichi pool, winners, deal-in hands and speed from events."""
    scores=[25000]*4;pool=0;hands=wins=hits=0;draws=0;won=hit=False;discount=0.;actions=Counter()
    for e in events:
        typ=e['type']
        if typ=='start_kyoku':
            if e['scores']!=scores or e['kyotaku']*1000!=pool:raise ValueError('Hand-boundary payments differ')
            draws=0;won=hit=False
        elif typ=='tsumo':draws+=1
        elif typ=='reach_accepted':scores[e['actor']]-=1000;pool+=1000
        elif typ in ('hora','ryukyoku'):
            deltas=e['deltas']
            if len(deltas)!=4 or sum(deltas)!=(pool if typ=='hora' else 0):raise ValueError('Four-seat payments do not conserve points')
            scores=[a+b for a,b in zip(scores,deltas)]
            if typ=='hora':
                pool=0
                if e['actor']==seat:
                    if won:raise ValueError('Duplicate own win within a hand')
                    won=True
                if e['actor']!=seat and e['target']==seat:hit=True
        elif typ=='end_kyoku':
            hands+=1;wins+=won;hits+=hit
            if won:discount+=gamma**draws
        elif typ=='end_game' and pool:
            leader=max(range(4),key=lambda p:(scores[p],-p));scores[leader]+=pool;pool=0
        if e.get('actor')==seat:actions[typ]+=1
        if sum(scores)+pool!=100000:raise ValueError('Point/pool conservation failed')
    if pool or not hands:raise ValueError('Incomplete match or unallocated terminal pool')
    rank=1+sum(scores[p]>scores[seat] or (scores[p]==scores[seat] and p<seat) for p in range(4))
    return dict(scores=scores,rank=rank,hands=hands,wins=wins,dealin_hands=hits,
        discount=discount,calls=actions['chi']+actions['pon'],
        kan_choices={k:actions[k] for k in ('ankan','daiminkan','kakan')})


def paired(values):
    from scipy.stats import t
    if len(values)<2:raise ValueError('Multiple independent wall groups required')
    mean=statistics.mean(values);half=float(t.ppf(.975,len(values)-1))*statistics.stdev(values)/math.sqrt(len(values))
    return dict(gain=mean,ci95=[mean-half,mean+half])


def independent_outcomes(rows,cfg):
    # These constants are deliberately independent of the evaluator's grid.
    suites=('closed','mixed','strong');policies=('reference','candidate')
    seeds=range(cfg['seed_start'],cfg['seed_start']+cfg['seeds'])
    grid={(s,n,p,k) for s in suites for n in seeds for p in range(4) for k in policies}
    indexed={(r['suite'],r['seed'],r['seat'],r['policy']):r for r in rows}
    if set(indexed)!=grid or len(indexed)!=len(rows):raise ValueError('Incomplete, duplicate or foreign paired grid')
    actual={}
    for key,row in indexed.items():
        m=recount(row['mjai_log'],row['seat'],cfg['fast_gamma']);actual[key]=m
        if (m['scores']!=row['final_scores'] or m['scores'][row['seat']]!=row['score']
                or m['rank']!=row['rank'] or m['hands']!=row['hands'] or m['wins']!=row['hand_wins']
                or abs(m['discount']-row['fast_reward_sum'])>1e-10 or m['calls']!=row['calls']
                or m['kan_choices']!=row['kan_choices']):raise ValueError('Actual event outcome or action count differs')
    output={}
    for name,field in [('hand_win_paired','wins'),('dealin_rate_paired','dealin_hands'),('fast_reward_paired','discount')]:
        differences=[]
        for n in seeds:
            ds=[]
            for s in suites:
                for p in range(4):
                    a,b=actual[s,n,p,'candidate'],actual[s,n,p,'reference'];ds.append(a[field]/a['hands']-b[field]/b['hands'])
            differences.append(statistics.mean(ds))
        output[name]=paired(differences)
    output['net_points_paired']=paired([statistics.mean(indexed[s,n,p,'candidate']['score']-indexed[s,n,p,'reference']['score']
        for s in suites for p in range(4)) for n in seeds])
    return output


def audit(cfg):
    from .metrics import summarize
    evaluation_file=Path(cfg['evaluation_config']).resolve()
    if digest(evaluation_file)!=cfg['evaluation_config_sha256']:raise ValueError('Registered evaluation recipe changed')
    ecfg=read(evaluation_file);out=Path(ecfg['out']);protocol=read(out/'protocol.json');verify(protocol)
    if protocol['config']!=ecfg:raise ValueError('Actual evaluator startup differs')
    candidate=checked_candidate(ecfg['candidate_run']);assert_same_candidate(protocol,candidate)
    verify_files(protocol['source_files']);verify_files(protocol['native_preflight_receipt']);verify_files(protocol.get('development_receipt',{}))
    native=read(out/'native-preflight.json')
    if native['candidate_identity_sha256']!=candidate['model']['identity_sha256'] or any(n<1 for n in native['routes'].values()):
        raise ValueError('Actual unified native coverage missing')
    result=read(out/'evaluation-result.json');raw=out/'games-private.jsonl';raw_sha=digest(raw)
    component='unified_continued_adapter_and_pointer_head' if candidate.get('initialization_mode')=='continued' else 'unified_new_adapter_and_pointer_head'
    if result.get('candidate_component')!=component:raise ValueError('Candidate initialization component differs')
    completion=read(out/'audit.json')
    if (completion['status']!='evaluation_complete_independent_audit_pending'
            or completion['games_sha256']!=raw_sha or completion['candidate_identity_sha256']!=candidate['model']['identity_sha256']):
        raise ValueError('Actual evaluation completion receipt differs')
    verify_files(completion['native_files'])
    rows=[json.loads(line) for line in raw.read_text().splitlines()]
    outcomes=independent_outcomes(rows,ecfg)
    expected=summarize(rows,ecfg)
    if any(result.get(k)!=v for k,v in expected.items()):raise ValueError('Evaluation aggregate differs')
    for name in ('hand_win_paired','dealin_rate_paired','fast_reward_paired'):
        m=outcomes[name];reported=result[name]
        if abs(m['gain']-reported['gain'])>1e-10 or any(abs(a-b)>1e-10 for a,b in zip(m['ci95'],reported['ci95'])):
            raise ValueError('Independent whole-wall hand/risk/speed interval differs')
    m=outcomes['net_points_paired'];reported=result['overall']['candidate_minus_reference']
    if abs(m['gain']-reported['gain'])>1e-9 or any(abs(a-b)>1e-9 for a,b in zip(m['ci95'],reported['ci95'])):
        raise ValueError('Independent whole-wall net-point interval differs')
    verified=(ecfg['stage']=='independent_confirmation' and outcomes['hand_win_paired']['ci95'][0]>0
        and outcomes['dealin_rate_paired']['gain']<=ecfg['max_dealin_rate_increase'])
    if digest(raw)!=raw_sha or bundle(candidate['model']['directory'])!=candidate['model']:
        raise ValueError('Model or games changed during independent audit')
    proof=dict(status='passed_complete',games=len(rows),games_sha256=raw_sha,
        candidate_identity_sha256=candidate['model']['identity_sha256'],selected_step=candidate['selected_step'],
        candidate_component=component,whole_wall_groups=ecfg['seeds'],
        manual_four_seat_payment_records=len(rows),manual_final_ranks_checked=len(rows),
        paired_outcomes=outcomes,reference_games_reused=0,independent_strength_improvement_verified=verified,
        han_fu_independently_recalculated=False,new_simulations=0,evaluation_config_sha256=digest(evaluation_file),
        unreviewed_evaluation_result_sha256=digest(out/'evaluation-result.json'))
    if (out/'independent-audit.json').exists() or (out/'result.json').exists():raise ValueError('Do not overwrite a completed independent audit')
    audit_out=Path(cfg['out']);audit_out.mkdir(parents=True,exist_ok=False)
    write(audit_out/'protocol.json',dict(config=cfg,evaluation_protocol_sha256=digest(out/'protocol.json'),
        candidate_bundle=candidate['model'],evaluation_result_sha256=proof['unreviewed_evaluation_result_sha256']))
    write(audit_out/'result.json',proof);write(audit_out/'status.json',proof)
    write(out/'independent-audit.json',proof)
    write(out/'result.json',dict(result,independent_audit_required=False,improvement_verified=verified))
    write(out/'status.json',dict(status='independently_audited_complete',games=len(rows),improvement_verified=verified))
    return proof


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    print(json.dumps(audit(read(a.config))))


if __name__=='__main__':main()
