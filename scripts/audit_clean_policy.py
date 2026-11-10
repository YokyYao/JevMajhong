"""Recompute paired full-policy scores and verify frozen source/weight identities."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from call_policy import digest
from clean_policy_evaluation import aggregate
from head_rl import write
from scripts.audit_guide_calls import compare


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());out=ROOT/cfg['out']
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol['config']!=cfg:raise ValueError('Evaluation config changed')
    for name,h in protocol['code_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Evaluation code changed: '+name)
    training_cfg=protocol['training_config'];run=ROOT/training_cfg['out'];reference=ROOT/training_cfg['reference']
    if digest(run/'result.json')!=protocol['training_result_sha256']:
        raise ValueError('Selected training result changed')
    for name,h in protocol['reference_files'].items():
        if digest(reference/name)!=h:raise ValueError('Frozen reference changed')
    for name,h in protocol['candidate_files'].items():
        if digest(ROOT/protocol['candidate_checkpoint']/name)!=h:raise ValueError('Frozen candidate changed')
    training=json.loads((run/'protocol.json').read_text())
    for name,h in training['code_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Training source changed')
    dev=set(range(training_cfg['development_seed_start'],training_cfg['development_seed_start']+training_cfg['development_seeds']))
    confirm=set(range(training_cfg['confirmation_seed_start'],training_cfg['confirmation_seed_start']+training_cfg['confirmation_seeds']))
    if dev&confirm:raise ValueError('Development and confirmation walls overlap')
    parity=json.loads((out/'native-parity.json').read_text())
    if parity['checked']!=9*cfg['parity_per_phase_view'] or not parity['decisions_equal'] or parity['max_absolute_probability_error']>cfg['parity_atol']:
        raise ValueError('Serving parity incomplete')
    rows_path=out/'games-private.jsonl';rows=[json.loads(l) for l in rows_path.read_text().splitlines()]
    actual=aggregate(rows,cfg);compare(json.loads((out/'summary.json').read_text()),actual)
    result={'status':'passed_complete','games':len(rows),'stage':cfg['stage'],
        'independent_wall_streams':cfg['seeds'],'overall_paired_gain':actual['overall']['candidate_minus_reference'],
        'raw_games_sha256':digest(rows_path),'candidate_frozen':True,'native_parity_passed':True,
        'source_identity_passed':True,'summary_recomputed':True,'human_test_opened':False,'deployed':False}
    write(out/'evaluation-audit.json',result);write(out/'status.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
