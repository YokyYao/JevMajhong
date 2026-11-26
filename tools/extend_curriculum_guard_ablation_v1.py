"""Reuse the first16 all-legal games; play only the remaining48 development games."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha


def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def extend(out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS') and not socket.gethostname().startswith(('lo-','login'))
    out.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'runs/curriculum-guard-ablation-v1/evaluation-v1'
    receipt=read(parent/'binding-private.json');files=dict(receipt['files'])
    if not all(sha(path)==digest for path,digest in files.items()):raise ValueError('Completed parent binding changed')
    assert read(parent/'status-private.json')=={'status':'complete','matches':32}
    assert read(parent/'summary-private.json')['status']=='frozen_v8_guard_ablation_complete_development_only'
    registered=ROOT/'runs/curriculum-basic-repair-v1/fit-plan/registered.json';reg=read(registered);c=reg['curriculum']
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.evaluation.metrics import hand_dealin_count,interval
    from jevmahjong.core.scoring import settlement_ledger
    from tools.evaluate_curriculum import play
    from tools.curriculum_all_legal_policy_v1 import AllLegalPolicyAgent
    assert package_digest(Path(reg['source'])/'jevmahjong')==reg['source_sha256']
    checkpoint=Path(c['reference_checkpoint']);identity=bundle(checkpoint)
    assert identity==receipt['checkpoint'] and identity['identity_sha256']==c['reference_identity']
    cache=ROOT/c['reference_evaluation_cache'];protocol=read(cache/'protocol.json')
    seeds=list(range(2026200000,2026200008));first=set(seeds[:2]);remaining=seeds[2:]
    assert protocol['checkpoint']==identity and protocol['seeds']==seeds and protocol['profiles']==c['opponent_profiles'] and protocol['seats']==4
    key=lambda r:(r['seed'],r['profile'],r['seat'])
    expected={(seed,profile,seat) for seed in seeds for profile in c['opponent_profiles'] for seat in range(4)}
    first_expected={k for k in expected if k[0] in first}
    def load(path):
        digest=sha(path)
        if str(path) in files and files[str(path)]!=digest:raise ValueError('A previously bound match file changed')
        files[str(path)]=digest
        return [json.loads(line) for line in path.open()]
    guarded=load(cache/'matches-private.jsonl');all_legal=load(parent/'all_legal-matches-private.jsonl')
    old_guarded=load(parent/'original_guard-matches-private.jsonl')
    assert len(guarded)==64 and {key(r) for r in guarded}==expected
    assert len(all_legal)==len(old_guarded)==16 and {key(r) for r in all_legal}=={key(r) for r in old_guarded}==first_expected
    indexed={key(r):r for r in guarded}
    fields=('final_scores','rank','net_points','hands','wins','dealin_hands','mjai_log')
    assert all(all(r[field]==indexed[key(r)][field] for field in fields) for r in old_guarded)
    for mode,rows in (('original_guard',old_guarded),('all_legal',all_legal)):
        assert aggregate_complete_matches(rows)==read(parent/'summary-private.json')['summaries'][mode]
    def verify(row):
        seat=row['seat'];ledger=settlement_ledger(row['mjai_log'],row['final_scores'])
        if (row['completed'] is not True or row['ledger_passed'] is not True or ledger['hands']!=row['hands']
            or row['net_points']!=row['final_scores'][seat]-25000
            or row['rank']!=sorted(range(4),key=lambda p:(-row['final_scores'][p],p)).index(seat)+1
            or row['wins']!=sum(e['type']=='hora' and e.get('actor')==seat for e in row['mjai_log'])
            or row['dealin_hands']!=hand_dealin_count(row['mjai_log'],seat)):
            raise ValueError('Actual native ledger/rank/hand metrics differ')
    for row in guarded+all_legal:verify(row)
    for p in (Path(__file__),ROOT/'docs/experiments/plans/curriculum-guard-extension-20261123.md',
              parent/'binding-private.json',parent/'summary-private.json',parent/'status-private.json'):
        files[str(p.resolve())]=sha(p)
    write(out/'registered-private.json',dict(status='remaining48_guard_ablation_registered_before_inference',files=files,
        checkpoint=identity,reused_all_legal_matches=16,cached_original_guard_matches=64,new_all_legal_matches=48,
        remaining_seeds=remaining,development_extension_after_inspection=True,final_reserved_grid_unused=True,
        job_id=os.environ['SLURM_JOB_ID'],node=socket.gethostname()))
    client=UnifiedPolicyClient(load_model(checkpoint));agent=AllLegalPolicyAgent(client);new=[]
    with (out/'all-legal64-matches-private.jsonl').open('x') as stream:
        for row in all_legal:stream.write(json.dumps(row,separators=(',',':'))+'\n')
        stream.flush()
        for seed in remaining:
            for profile in c['opponent_profiles']:
                for seat in range(4):
                    row=play(agent,seed,seat,profile);verify(row);new.append(row);all_legal.append(row)
                    stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush()
                    (out/'status-private.json').write_text(json.dumps(dict(status='running',completed_new=len(new),total_new=48))+'\n')
    assert len(new)==48 and len(all_legal)==64 and {key(r) for r in all_legal}==expected
    after={key(r):r for r in all_legal}
    functions=dict(net_points=lambda r:r['net_points'],top2_rate=lambda r:float(r['rank']<=2),
        hand_win_rate=lambda r:r['wins']/r['hands'],dealin_hand_rate=lambda r:r['dealin_hands']/r['hands'])
    def comparison(selected_seeds):
        per_seed={str(seed):{field:statistics.mean(fn(after[k])-fn(indexed[k]) for k in expected if k[0]==seed)
            for field,fn in functions.items()} for seed in selected_seeds}
        return dict(seed_groups=len(selected_seeds),per_seed=per_seed,
            paired={field:dict(mean=statistics.mean(values[field] for values in per_seed.values()),
                exploratory_ci95=interval([values[field] for values in per_seed.values()])) for field in functions},
            independent_final_confirmation=False)
    if bundle(checkpoint)!=identity or not all(sha(path)==digest for path,digest in files.items()):raise ValueError('Frozen source/model/cache changed')
    result=dict(status='frozen_v8_guard_extension64_complete_development_only',
        summaries=dict(original_guard=aggregate_complete_matches(guarded),all_legal=aggregate_complete_matches(all_legal)),
        all8_seed_comparison=comparison(seeds),remaining6_seed_comparison=comparison(remaining),
        first16_result_preserved=read(parent/'summary-private.json'),new_all_legal_matches=48,reused_all_legal_matches=16,
        cached_original_guard_matches=64,shadow_counters_new48=dict(agent.shadow),
        shadow_counters_all64=dict(Counter(read(parent/'summary-private.json')['shadow_guard_counters']['all_legal'])+agent.shadow),
        checkpoint_identity=identity['identity_sha256'],
        extension_registered_after_first2_seed_results=True,no_independent_strength_or_deployment_claim=True,
        no_model_updates_or_active_policy_changes=True,final_reserved_grid_unused=True,raw_logs_private=True)
    write(out/'summary-private.json',result);write(out/'binding-private.json',dict(files=files,checkpoint=identity,job_id=os.environ['SLURM_JOB_ID']))
    (out/'status-private.json').write_text(json.dumps(dict(status='complete',new_matches=48,total_all_legal=64))+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();extend(args.out.resolve())
