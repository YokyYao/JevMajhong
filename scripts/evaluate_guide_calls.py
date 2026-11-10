"""Frozen guide-point candidate: native parity, then full-East development."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from arena import EfficiencyAgent
from call_policy import CallRouter,digest,request_hash
from guide_call_value import PointValueCallRouter,point_features,utility_preferences
from guide_development_audit import aggregate,SUITES,POLICIES
from head_rl import write
from riichienv import GameRule,RiichiEnv
from scoring_audit import settlement_ledger
from scripts.evaluate_call_repair import Agent,CallingOpponent
from scripts.train_call_repair import make_head
from seed_control import engine_seed
from semantic import build_request

CODE = ['scripts/evaluate_guide_calls.py','scripts/audit_guide_calls.py','guide_development_audit.py',
        'guide_call_value.py','guide_call_features.py','call_policy.py','semantic.py','arena.py',
        'scoring_audit.py','seed_control.py','scripts/evaluate_call_repair.py','scripts/train_call_repair.py']


def game(agent, strong, seed, seat, suite):
    env = RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(seed,'spaced-v1'))
    obs = env.reset(); ordinary,caller = EfficiencyAgent(),CallingOpponent(); steps = 0
    while not env.done():
        selected = {}
        for p,o in obs.items():
            policy = agent if p==seat else (strong if suite=='strong' else caller) if p==(seat+1)%4 and suite!='closed' else ordinary
            selected[p] = policy.act(o)
        obs = env.step(selected); steps += 1
        if steps > 100000: raise RuntimeError('Game failed to terminate')
    log = env.mjai_log; ledger = settlement_ledger(log,env.scores())
    actions = Counter(e['type'] for e in log if e.get('actor')==seat)
    return {'suite':suite,'seed':seed,'seat':seat,'score':env.scores()[seat],'rank':env.ranks()[seat],
        'hands':ledger['hands'],'hand_wins':sum(e['type']=='hora' and e['actor']==seat for e in log),
        'dealins':sum(e['type']=='hora' and e['target']==seat and e['actor']!=seat for e in log),
        'calls':actions['chi']+actions['pon'],'steps':steps,'point_components':ledger['components'][seat],
        'settlement_ledger_passed':True}


def native_parity(client,router,cpu_head,training_run,count):
    seen = set(); checked = 0
    for shard in sorted((training_run/'walls-private').glob('*.pt')):
        payload = torch.load(shard,map_location='cpu',weights_only=False)
        for row in payload['rows']:
            request = row['public_request']; key = request_hash(request)
            if key in seen: continue
            seen.add(key); result = router.system_one(**request); captured = router.last
            cpu = point_features(cpu_head,captured['decide'],captured['options'],request)
            values = cpu @ router.coefficient.cpu()
            if not torch.allclose(values,captured['values'],atol=2e-4,rtol=2e-4):
                raise ValueError('CPU/native point utilities differ')
            _, chosen = utility_preferences(values,captured['pass_index'])
            keys = list(request['questions']['action']['criteria'])
            p = result['answers']['action']['probabilities']
            if chosen != captured['chosen'] or keys[chosen] != max(keys,key=lambda k:p[k]):
                raise ValueError('CPU/native point decisions differ')
            checked += 1
            if checked >= count: break
        if checked >= count: break
    if checked < count: raise ValueError('Insufficient distinct training requests for native parity')
    # One already-used training start; ordinary discard path must remain exact.
    cfg = json.loads((training_run/'protocol.json').read_text())['config']
    env = RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(cfg['training_seed_start'],'spaced-v1'))
    obs = env.reset(); request = build_request(next(iter(obs.values())))[0]
    before = client.system_one(**copy.deepcopy(request))['answers']['action']['probabilities']
    after = router.system_one(**copy.deepcopy(request))['answers']['action']['probabilities']
    if before != after: raise ValueError('Ordinary discard native probabilities changed')
    return {'distinct_call_requests':checked,'cpu_native_value_and_decision_parity':True,
            'native_teacher_probability_checks':router.checks,'ordinary_discard_probabilities_unchanged':True,
            'source':'opened training requests only; no development or human test cases'}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    cfg = json.loads(args.config.read_text()); out = ROOT/cfg['out'];out.mkdir(parents=True,exist_ok=False)
    train_cfg = json.loads((ROOT/cfg['source_config']).read_text()); run = ROOT/train_cfg['out']
    fit = json.loads((run/'fit-result.json').read_text())
    if fit['status']!='point_predictor_fit':
        write(out/'development-skipped.json',{'reason':fit['status'],'deployed':False});return
    checkpoint = run/'point-value-private.pt'
    if digest(checkpoint)!=fit['checkpoint_sha256']:raise ValueError('Point predictor changed')
    source = json.loads((run/'protocol.json').read_text()); reference = ROOT/train_cfg['reference']
    for name,h in source['reference_files'].items():
        if digest(reference/name)!=h:raise ValueError('Frozen reference changed')
    if digest(ROOT/train_cfg['call_checkpoint'])!=train_cfg['call_head_sha256']:raise ValueError('Teacher changed')
    for name,h in source['code_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Training code changed before evaluation')
    protocol = {'config':cfg,'training_config':train_cfg,'point_checkpoint_sha256':digest(checkpoint),
        'fit_result_sha256':digest(run/'fit-result.json'),'reference_files':source['reference_files'],
        'code_sha256':{n:digest(ROOT/n) for n in CODE},'human_test_opened':False,'candidate_frozen':True,
        'opponents':{'closed':'3 EfficiencyAgent','mixed':'2 EfficiencyAgent +1 CallingOpponent at focal+1',
                     'strong':'2 EfficiencyAgent +1 frozen incumbentB at focal+1'},
        'primary':'three suites equally weighted per wall, then Student-t CI across32 logical walls',
        'not_real_player_win_rate':True}
    write(out/'protocol.json',protocol)
    from jevany import JevModel
    from jevany.checkpoint import LoadOptions
    from jevany.inference import InferenceOptions
    base = Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'
    write(out/'status.json',{'status':'loading_frozen_models','games':0})
    client = JevModel.from_pretrained(str(reference),device='cuda',dtype='bf16',
        options=LoadOptions(merge=False,temperature=1.,compile_mode=None,cuda_graphs=False,fused_kernels=False,base_load_path=str(base)),
        inference_options=InferenceOptions(max_state_tokens=4096,max_branch_tokens=8192,max_packed_tokens=8192,prefix_cache_size=0))
    meta = torch.load(reference/'head.pt',map_location='cpu',weights_only=True)
    call = torch.load(ROOT/train_cfg['call_checkpoint'],map_location='cpu',weights_only=True)
    point = torch.load(checkpoint,map_location='cpu',weights_only=True)
    if point['teacher_sha256']!=train_cfg['call_head_sha256'] or point['reference_sha256']!=train_cfg['reference_head_sha256']:
        raise ValueError('Point-predictor source identities differ')
    teacher = make_head(meta,'cuda');teacher.load_state_dict(call['head']);teacher.eval().requires_grad_(False)
    cpu_head = make_head(meta,'cpu');cpu_head.load_state_dict(call['head']);cpu_head.eval().requires_grad_(False)
    router = PointValueCallRouter(client,teacher,point)
    try:
        parity = native_parity(client,router,cpu_head,run,cfg['native_parity_requests']);write(out/'native-parity.json',parity)
        agents = {'reference':Agent(client),'previous':Agent(CallRouter(client,teacher)),'candidate':Agent(router)}
        strong = Agent(client); rows = []; started=time.monotonic()
        with (out/'games-private.jsonl').open('w') as stream:
            for suite in SUITES:
                for seed in range(cfg['development_seed_start'],cfg['development_seed_start']+cfg['development_seeds']):
                    for seat in range(4):
                        for name in POLICIES:
                            row = {'policy':name,**game(agents[name],strong,seed,seat,suite)};rows.append(row)
                            stream.write(json.dumps(row)+'\n');stream.flush()
                            status = {'status':'guide_point_development','suite':suite,'games':len(rows),
                                'total_games':cfg['development_seeds']*4*len(SUITES)*len(POLICIES),
                                'native_point_checks':router.checks,'seconds':time.monotonic()-started}
                            write(out/'status.json',status);print(json.dumps(status),flush=True)
        write(out/'summary.json',aggregate(rows,cfg))
        write(out/'status.json',{'status':'development_games_complete_audit_pending','games':len(rows),'deployed':False})
    finally:router.close()


if __name__=='__main__':main()
