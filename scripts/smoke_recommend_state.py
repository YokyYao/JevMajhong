"""Run the public recommendation CLI on one genuine, player-visible observation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from riichienv import RiichiEnv,GameRule
from semantic import build_request
from arena import guarded_indices
from seed_control import engine_seed
from head_rl import HeadPolicy


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('Preserve previous interface smoke')
    initial_head_sha=hashlib.sha256((args.checkpoint/'head.pt').read_bytes()).hexdigest()
    env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(3401000,'spaced-v1'))
    obs=env.reset()[0]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    exported=args.out.with_suffix('.observation.txt');exported.write_text(obs.serialize_to_base64()+'\n')
    result=subprocess.run([sys.executable,str(ROOT/'recommend_state.py'),str(exported),
                           '--checkpoint',str(args.checkpoint),'--guarded'],cwd=ROOT,
                          capture_output=True,text=True,check=True)
    payload=json.loads(result.stdout.strip().splitlines()[-1])
    _,actions,features=build_request(obs);eligible=guarded_indices(obs,actions,features)
    chosen=json.dumps(payload['action'],sort_keys=True)
    allowed={json.dumps(json.loads(actions[i].to_mjai()),sort_keys=True) for i in eligible}
    if payload['policy']!='model' or payload['guarded'] is not True or chosen not in allowed:
        raise ValueError('Public recommendation CLI returned a policy/action outside the evaluated guard')
    if len(eligible)<2:
        raise ValueError('The interface smoke must exercise a nontrivial guarded choice')
    evaluated=HeadPolicy({'reference':str(args.checkpoint),'seed':3401000,
                          'advantage_estimator':'greedy_reference','exploration_uniform':.15,
                          'reward_mode':'net_points'})
    expected=json.dumps(json.loads(evaluated.act(obs).to_mjai()),sort_keys=True)
    if expected!=chosen or evaluated.checks<1:
        raise ValueError('Public recommendation differs from the evaluated native head policy')
    if hashlib.sha256((args.checkpoint/'head.pt').read_bytes()).hexdigest()!=initial_head_sha:
        raise ValueError('Inference modified the source checkpoint')
    record={'status':'passed','checkpoint':str(args.checkpoint),'head_sha256':hashlib.sha256((args.checkpoint/'head.pt').read_bytes()).hexdigest(),
            'real_player_observation':True,'legal_guarded_action_returned':True,'public_cli_exercised':True,
            'matches_evaluated_head_policy_action':True,'eligible_actions':len(eligible),
            'source_checkpoint_unchanged':True,
            'runtime':'CUDA bf16 unmerged temperature1, same context bounds as evaluation',
            'performance_claim':False,'code_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
              for n in ['native_policy.py','recommend_state.py','scripts/smoke_recommend_state.py']}}
    args.out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
