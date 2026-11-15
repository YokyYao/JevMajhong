"""Capture non-threat ordinary decisions in closed and opened hands."""
import argparse
import json
from pathlib import Path
from riichienv import GameRule,RiichiEnv
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import build_request
from jevmahjong.data.native_states import ATTACK_METHOD
from jevmahjong.policy.ordinary import is_attack_request,request_eligible
from .collector import NativeCollector,call_observation_digest
from .defense_states import collect_native_inputs
from .futures import replay_step
from .rewards import hand_outcomes


class AttackStateCollector(NativeCollector):
    def act(self,obs,frozen=False,stochastic=False,trace=None):
        captured=[] if trace is not None else None
        chosen=super().act(obs,frozen,stochastic,captured)
        if captured:
            for row in captured:
                if row['scope']=='ordinary' and is_attack_request(row['public_request']):
                    if row['eligible']!=request_eligible(row['public_request']):
                        raise ValueError('Public and native attacking admission differ')
                    trace.append(row)
        return chosen


def enumerate_control(control,config):
    """Replay every control step and independently enumerate admissible inputs."""
    seed,seat=control['seed'],control['seat']
    env=RiichiEnv(game_mode=config['game_mode'],rule=GameRule.default_mjsoul(),
        seed=engine_seed(seed,config['seed_scheme']))
    observations=env.reset();log=control['mjai_log'];step=0;decisions=[]
    if env.mjai_log!=log[:len(env.mjai_log)]:raise ValueError('Attacking control initial prefix differs')
    while not env.done():
        if seat in observations:
            request=build_request(observations[seat])[0]
            if is_attack_request(request) and len(request_eligible(request))>1:
                hands=hand_outcomes(env.mjai_log);key=(seed,seat,len(decisions))
                decisions.append(dict(episode_key=key,seed=seed,seat=seat,replica=key[2],
                    intervention_step=step,intervention_state_digest=call_observation_digest(observations),
                    hand_index=len(hands)-1,draws_before=hands[-1]['draws']))
        previous=len(env.mjai_log);observations=replay_step(env,observations,log,step);step+=1
        if env.mjai_log[previous:]!=log[previous:len(env.mjai_log)]:
            raise ValueError('Attacking control prefix differs')
        if step>control['environment_steps']:raise ValueError('Attacking control exceeds saved steps')
    if step!=control['environment_steps'] or env.mjai_log!=log or env.scores()[seat]!=control['score']:
        raise ValueError('Attacking control terminal replay differs')
    return decisions


def audit_control_coverage(payload,config):
    expected=[]
    for control in payload['controls']:expected.extend(enumerate_control(control,config))
    if expected!=payload['decisions']:raise ValueError('Attacking control opportunity coverage differs')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--max-walls',type=int)
    args=parser.parse_args()
    collect_native_inputs(json.loads(args.config.read_text()),args.resume,args.max_walls,
        collector=AttackStateCollector,method=ATTACK_METHOD,scope='ordinary')


if __name__=='__main__':main()
