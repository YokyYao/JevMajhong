"""Native stage4 accept/decline routing checks; no labels, fit or release."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class MenuSelector:
    def __init__(self,kind):self.kind=kind;self.calls=0;self.last_request=None
    def system_one(self,**request):
        self.calls+=1;self.last_request=request
        criteria=request['questions']['action']['criteria']
        keys=[k for k,v in criteria.items() if v['type']==self.kind]
        assert keys,'Requested native operation unavailable'
        key=keys[0]
        return dict(answers=dict(action=dict(probabilities={k:float(k==key) for k in criteria})))


def audit(registered,out,seed_start=2026400100,games=24):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-','login'))
    assert games>0 and not set(range(seed_start,seed_start+games))&set(range(2026300000,2026300032))
    reg=json.loads(Path(registered).read_text());sys.path.insert(0,reg['source'])
    sys.path.insert(0,reg['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from riichienv import ActionType,RiichiEnv,GameRule
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.state import legal_candidates,build_request,signature
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.agents import EfficiencyAgent,PolicyAgent
    from tools.curriculum_endgame import EndgamePolicyAgent,in_endgame_scope,public_placement,SCOPE
    assert package_digest(Path(reg['source'])/'jevmahjong')==reg['source_sha256']
    totals=Counter();receipts=[];private=[]
    for seed in range(seed_start,seed_start+games):
        env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(seed,'spaced-v1'))
        observations=env.reset();baseline=EfficiencyAgent();declined=set();counts=Counter()
        for step in range(30000):
            if env.done():break
            selected={}
            for seat,obs in observations.items():
                actions=legal_candidates(obs);wins=[a for a in actions if a.action_type in (ActionType.RON,ActionType.TSUMO)]
                if not wins or len(actions)==1:selected[seat]=baseline.act(obs);continue
                request,encoded,_=build_request(obs);state=request['state'];late=in_endgame_scope(state)
                kind='ron' if wins[0].action_type==ActionType.RON else 'tsumo'
                alternate='none' if kind=='ron' else 'dahai'
                available=any(c['type']==alternate for c in request['questions']['action']['criteria'].values())
                decline=late and available and kind not in declined
                selector=MenuSelector(alternate if decline else 'hora');agent=EndgamePolicyAgent(selector)
                choice=agent.act(obs)
                assert signature(choice) in {signature(a) for a in actions}
                if late:
                    assert selector.calls==1 and agent.win_windows==1
                    assert set(selector.last_request)=={'state','questions'}
                    assert set(selector.last_request['questions']['action'])=={'type','instructions','criteria'}
                    assert selector.last_request==request
                    assert not any(statekey in state for statekey in ('wall','opponent_hands','future'))
                    counts['late_'+kind+'_model_windows']+=1
                    counts['late_start_rank_'+str(public_placement(state)['rank'])]+=1
                    counts['late_dealer_win_windows']+=seat==state['dealer']
                    counts['extension_win_windows']+=state['round_wind']>0
                    if decline:
                        assert choice.action_type==(ActionType.PASS if kind=='ron' else ActionType.DISCARD)
                        assert agent.win_declines==1;declined.add(kind);counts[kind+'_declines']+=1
                    else:
                        assert choice.action_type in (ActionType.RON,ActionType.TSUMO)
                        assert agent.win_declines==0;counts[kind+'_model_accepts']+=1
                    # Reference takes every offered legal win without consulting its model.
                    reference_selector=MenuSelector('none');reference=PolicyAgent(reference_selector).act(obs)
                    assert reference.action_type in (ActionType.RON,ActionType.TSUMO) and reference_selector.calls==0
                    private.append(dict(seed=seed,step=step,seat=seat,kind=kind,public_request=request,
                        selection=json.loads(choice.to_mjai()),early_reference_guard_preserved=True))
                else:
                    assert selector.calls==0 and choice.action_type in (ActionType.RON,ActionType.TSUMO)
                    counts['early_forced_wins_without_model']+=1
                selected[seat]=choice
            observations=env.step(selected)
        assert env.done(),'Native endgame routing truncated a match'
        ledger=settlement_ledger(env.mjai_log,env.scores())
        receipts.append(dict(seed=seed,completed=True,hands=ledger['hands'],counts=dict(counts)))
        totals.update(counts)
    covered=all(totals.get(k,0)>0 for k in ('ron_declines','tsumo_declines','ron_model_accepts','tsumo_model_accepts','early_forced_wins_without_model'))
    result=dict(status='stage4_native_model_route_checked' if covered else 'stage4_native_route_coverage_incomplete',
        node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],seed_start=seed_start,games=games,
        scope=SCOPE,totals=dict(totals),aggregate_receipts=receipts,coverage_passed=covered,
        one_model_call_per_late_win_window=True,early_and_practical_v8_guard_unchanged=True,
        real_model_loaded=False,scripted_selections_not_training_labels=True,stage4_fit_admitted=False,
        active_fits_modified=False,unseen_final_seeds_not_used=True,internal_ledgers_passed=True,
        independent_han_fu_scoring=False)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_name(out.stem+'-windows-private.jsonl').open('x') as f:
        for row in private:f.write(json.dumps(row)+'\n')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--seed-start',type=int,default=2026400100)
    p.add_argument('--games',type=int,default=24);a=p.parse_args();audit(a.registered,a.out,a.seed_start,a.games)
