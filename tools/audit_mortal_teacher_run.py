"""Independent accounting/statistics audit of a finished teacher experiment.

Does not call the evaluator's ledger or summarizer. This checks payments,
pool accounting, terminal ranks and paired statistics, not han/fu rules.
Run only inside a scheduled GPU allocation, including CPU-side auditing.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def account(log, final_scores):
    if not log or log[0]['type']!='start_game' or log[-1]['type']!='end_game':
        raise ValueError('Require an actual complete native game log')
    scores=[25000]*4;pool=0;hands=0;open_hand=False;ended=False;hit=set()
    wins=Counter();deal_events=Counter();deal_hands=Counter();choices={p:Counter() for p in range(4)}
    for e in log:
        kind=e['type']
        if ended: raise ValueError('Events after terminal game')
        if kind=='start_kyoku':
            if open_hand or list(e['scores'])!=scores or e['kyotaku']*1000!=pool:
                raise ValueError('Independent start-hand payment/pool mismatch')
            hands+=1;open_hand=True;hit=set()
        elif kind=='reach_accepted':
            if not open_hand: raise ValueError('Deposit outside hand')
            scores[e['actor']]-=1000;pool+=1000
        elif kind in ('hora','ryukyoku'):
            delta=e['deltas']
            if not open_hand or len(delta)!=4 or any(type(v) is not int for v in delta):
                raise ValueError('Invalid native payment vector')
            if sum(delta)!=(pool if kind=='hora' else 0):
                raise ValueError('Independent win/draw pool conservation failed')
            scores=[a+b for a,b in zip(scores,delta)]
            if kind=='hora':
                pool=0;wins[e['actor']]+=1
                if e['target']!=e['actor']:
                    hit.add(e['target']);deal_events[e['target']]+=1
        elif kind=='end_kyoku':
            if not open_hand: raise ValueError('Hand ended twice')
            for p in hit: deal_hands[p]+=1
            open_hand=False
        elif kind=='end_game':
            if open_hand or not hands: raise ValueError('Terminal game before hand completion')
            if pool:
                leader=min(range(4),key=lambda p:(-scores[p],p))
                scores[leader]+=pool;pool=0
            ended=True
        if kind in ('chi','pon','ankan','daiminkan','kakan','reach'):
            choices[e['actor']][kind]+=1
        if sum(scores)+pool!=100000:
            raise ValueError('Independent per-event points conservation failed')
    if not ended or pool or scores!=list(final_scores):
        raise ValueError('Independent terminal score mismatch')
    order=sorted(range(4),key=lambda p:(-scores[p],p));ranks=[order.index(p)+1 for p in range(4)]
    return dict(final_scores=scores,ranks=ranks,hands=hands,wins=wins,
                deal_events=deal_events,deal_hands=deal_hands,choices=choices)


def near(a,b):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or abs(a-b)>1e-10:
        raise ValueError(f'Independent summary mismatch: {a} versus {b}')


def audit_games(rows, seeds):
    from scipy.stats import t
    expected={(a,n,p,s) for a in ('parent','candidate') for n in seeds for p in ('calling','mortal') for s in range(4)}
    keys=[(r['arm'],r['seed'],r['profile'],r['seat']) for r in rows]
    if len(keys)!=len(expected) or len(set(keys))!=len(keys) or set(keys)!=expected:
        raise ValueError('Incomplete/duplicate independent paired game grid')
    values=[]
    for r in rows:
        if not r['completed'] or not r['ledger_passed']: raise ValueError('Evaluator reported an incomplete game')
        x=account(r['mjai_log'],r['final_scores']);p=r['seat']
        if (x['ranks'][p]!=r['rank'] or x['final_scores'][p]-25000!=r['net_points']
                or x['hands']!=r['hands'] or x['wins'][p]!=r['wins'] or x['deal_events'][p]!=r['dealins']):
            raise ValueError('Saved outcomes differ from independent log recount')
        if dict(x['choices'][p])!={k:v for k,v in r['choices'].items() if v}:
            raise ValueError('Saved call/kan/riichi count differs')
        values.append(dict(arm=r['arm'],seed=r['seed'],rank=x['ranks'][p],points=x['final_scores'][p]-25000,
            hands=x['hands'],wins=x['wins'][p],deal_events=x['deal_events'][p],deal_hands=x['deal_hands'][p],
            choices=x['choices'][p]))
    stats={}
    for arm in ('parent','candidate'):
        v=[r for r in values if r['arm']==arm];hands=sum(r['hands'] for r in v)
        stats[arm]=dict(games=len(v),top2_rate=statistics.mean(r['rank']<=2 for r in v),
            first_rate=statistics.mean(r['rank']==1 for r in v),fourth_rate=statistics.mean(r['rank']==4 for r in v),
            mean_rank=statistics.mean(r['rank'] for r in v),mean_net_points=statistics.mean(r['points'] for r in v),
            hand_win_rate=sum(r['wins'] for r in v)/hands,
            dealin_event_rate=sum(r['deal_events'] for r in v)/hands,
            dealin_hand_rate=sum(r['deal_hands'] for r in v)/hands,
            choices=dict(sum((r['choices'] for r in v),Counter())))
    differences=[statistics.mean(int(r['rank']<=2) for r in values if r['seed']==seed and r['arm']=='candidate')
        -statistics.mean(int(r['rank']<=2) for r in values if r['seed']==seed and r['arm']=='parent') for seed in seeds]
    gain=statistics.mean(differences);half=float(t.ppf(.975,len(seeds)-1))*statistics.stdev(differences)/math.sqrt(len(seeds))
    return dict(arms=stats,top2_gain=gain,whole_seed_ci95=[gain-half,gain+half],seed_groups=len(seeds),
        checked_complete_games=len(rows),independent_confirmation=False,
        accounting_not_independent_han_fu=True,release_qualified=False)


def audit(c):
    from tools.build_mortal_teacher import sha,verify
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.futures import bundle_identity
    out=ROOT/c['out'];read=lambda p:json.loads(Path(p).read_text())
    verify(read(out/'audit-registration-private.json'))
    fit=read(out/'fit-result-private.json');model_audit=read(out/'audit-private.json')
    pipeline=read(out/'pipeline-exit-private.json');protocol=read(out/'comparison-protocol-private.json')
    status=read(out/'comparison-status-private.json');saved=read(out/'comparison-result-private.json')
    if pipeline['exit_code']!=0 or fit['status']!='mortal_teacher_fit_complete' or status['status']!='complete':
        raise ValueError('Actual pipeline/comparison not complete')
    training_proof=read(out/'training-protocol-private.json');verify(training_proof)
    verify({'files':training_proof['data_files']});verify({'files':training_proof['old_validation_files']})
    if fit['last_new_update']<c['observation_new_updates'] or fit['last_new_update']>c['max_new_updates']:
        raise ValueError('Registered observation/max duration was not satisfied')
    updates=[json.loads(s) for s in (out/'updates-private.jsonl').read_text().splitlines()]
    if [x['new_update'] for x in updates]!=list(range(1,fit['last_new_update']+1)):
        raise ValueError('Missing actual optimizer updates')
    if any(x['source_counts']!=c['batch_quotas'] for x in updates) or updates[-1]['unique_exposure']!=1898:
        raise ValueError('Fixed source mixture or full exposure not satisfied')
    for x in fit['curve']:
        if x['validation_path']!='actual_saved_normal_serving' or x['total_step']!=176+x['new_update']:
            raise ValueError('Validation is not actual saved normal serving')
        if x['new_update'] and not(x['lora_changed'] and x['head_changed']):
            raise ValueError('Existing LoRA and head did not both update')
    audited=next(x for x in model_audit['audits'] if x['total_step']==protocol['selected_total_step'])
    logged=next(x for x in fit['curve'] if x['total_step']==audited['total_step'])
    if audited['metrics']!=logged['metrics']:
        raise ValueError('Selected normal-load reproduction differs')
    for arm,identity in protocol['arms'].items():
        path=ROOT/c['parent_checkpoint'] if arm=='parent' else out/'training-run/checkpoints'/f"step-{audited['total_step']:06d}"
        if bundle_identity(path)!=identity: raise ValueError('Comparison model identity changed')
    rows=[json.loads(s) for s in (out/'comparison-games-private.jsonl').read_text().splitlines()]
    expected_seeds=list(range(c['comparison_seed_start'],c['comparison_seed_start']+c['comparison_seed_groups']))
    if protocol['seeds']!=expected_seeds: raise ValueError('Comparison seeds differ')
    report=audit_games(rows,expected_seeds)
    for arm,metrics in report['arms'].items():
        for key,value in metrics.items():
            if key=='dealin_hand_rate': continue
            old_key='dealin_rate' if key=='dealin_event_rate' else key
            if key=='choices':
                if value!=saved['arms'][arm][key]: raise ValueError('Summary action counts differ')
            else: near(value,saved['arms'][arm][old_key])
    near(report['top2_gain'],saved['top2_gain'])
    for a,b in zip(report['whole_seed_ci95'],saved['whole_seed_ci95']): near(a,b)
    source_paths=list((out/'sources').glob('*.json'))
    if len(source_paths)!=512: raise ValueError('Source-game count differs')
    for path in source_paths:
        source=read(path);account(source['mjai_log'],source['final_scores'])
    report.update(status='independent_accounting_and_statistics_passed',source_games_checked=len(source_paths),
        comparison_games_sha256=sha(out/'comparison-games-private.jsonl'),auditor_sha256=sha(__file__),
        selected_total_step=audited['total_step'],screen_eligible=protocol['screen_eligible'],
        comparison_reason=protocol['comparison_reason'],raw_publication=False)
    verify(read(out/'audit-registration-private.json'));write(out/'independent-audit-private.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    from tools.build_mortal_teacher import require_gpu
    require_gpu();c=json.loads(a.config.read_text());sys.path.insert(0,c['upstream_source']);audit(c)
