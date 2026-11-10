"""Recompute the complete guide-point development result from private games."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from call_policy import digest
from guide_development_audit import aggregate
from head_rl import write


def compare(recorded,actual,path='summary'):
    if isinstance(actual,dict):
        if not isinstance(recorded,dict) or set(recorded)!=set(actual):raise ValueError(f'{path} keys differ')
        for k,v in actual.items():compare(recorded[k],v,path+'.'+k)
    elif isinstance(actual,list):
        if not isinstance(recorded,list) or len(recorded)!=len(actual):raise ValueError(f'{path} length differs')
        for i,v in enumerate(actual):compare(recorded[i],v,path+f'[{i}]')
    elif isinstance(actual,(int,float)) and not isinstance(actual,bool):
        if not isinstance(recorded,(int,float)) or not math.isclose(recorded,actual,rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError(f'{path} differs from raw games')
    elif recorded!=actual:raise ValueError(f'{path} differs')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    cfg=json.loads(args.config.read_text());out=ROOT/cfg['out'];protocol=json.loads((out/'protocol.json').read_text())
    if protocol['config']!=cfg:raise ValueError('Development config changed')
    for name,h in protocol['code_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Development code changed: '+name)
    source_cfg=protocol['training_config'];training=ROOT/source_cfg['out'];reference=ROOT/source_cfg['reference']
    for name,h in protocol['reference_files'].items():
        if digest(reference/name)!=h:raise ValueError('Reference changed')
    if digest(training/'point-value-private.pt')!=protocol['point_checkpoint_sha256']:raise ValueError('Point predictor changed')
    if digest(training/'fit-result.json')!=protocol['fit_result_sha256']:raise ValueError('Training result changed')
    if digest(ROOT/source_cfg['call_checkpoint'])!=source_cfg['call_head_sha256']:raise ValueError('Teacher changed')
    training_seeds=set(range(source_cfg['training_seed_start'],source_cfg['training_seed_start']+source_cfg['training_seeds_per_stage']))
    development_seeds=set(range(cfg['development_seed_start'],cfg['development_seed_start']+cfg['development_seeds']))
    confirmation_seeds=set(range(cfg['confirmation_seed_start'],cfg['confirmation_seed_start']+cfg['confirmation_seeds']))
    if training_seeds&development_seeds or training_seeds&confirmation_seeds or development_seeds&confirmation_seeds:
        raise ValueError('Train/development/confirmation wall overlap')
    parity=json.loads((out/'native-parity.json').read_text())
    if parity['distinct_call_requests']!=cfg['native_parity_requests'] or not parity['cpu_native_value_and_decision_parity'] or not parity['ordinary_discard_probabilities_unchanged']:
        raise ValueError('Native point interface parity incomplete')
    path=out/'games-private.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()]
    actual=aggregate(rows,cfg);compare(json.loads((out/'summary.json').read_text()),actual)
    result={'status':'passed_complete','games':len(rows),'wall_clusters':cfg['development_seeds'],
        'raw_games_sha256':digest(path),'candidate_sha256':protocol['point_checkpoint_sha256'],
        'native_parity_passed':True,'source_identity_passed':True,'summary_recomputed':True,
        'overall_paired_gain':actual['overall']['candidate_minus_reference'],
        'not_final_confirmation':True,'deployed':False}
    write(out/'development-audit.json',result);write(out/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
