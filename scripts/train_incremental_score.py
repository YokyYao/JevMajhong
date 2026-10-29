"""Collect fresh batches and continue point-ranking without a fixed-gain veto."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from head_rl import fingerprint,write
from incremental_score import IncrementalFrozenTeacherPolicy,incremental_gate,select_incremental
from point_pairwise import PointPairwiseTrainer
from scripts.train_score_pairwise import pairwise_checkpoint
from scripts.train_single_decision import intervention_advantages
from scripts import train_head_rl as runner


def protocol_write(path,data):
    if Path(path).name=='protocol.json':
        data=copy.deepcopy(data)
        names=['incremental_score.py','scripts/train_incremental_score.py','point_pairwise.py',
               'single_decision_rl.py','scripts/train_score_pairwise.py','scripts/train_calibrated_cache.py',
               'scripts/train_single_decision.py','reference_calibration.py',
               'scripts/audit_score_confirmation.py','scripts/prepare_incremental_score.py',
               'scripts/prepare_positive_confirmation.py','semantic.py','seed_control.py']
        data['code_sha256'].update({n:fingerprint(ROOT/n) for n in names})
        data.update(training_objective='point_delta_weighted_reference_relative_pairwise_ranking',
                    ppo_surrogate_used=False,teacher_control_stationary_across_stages=True,
                    fixed_gain_gate_enabled=False,selection_policy='best_positive_complete_development')
    write(path,data)


if __name__=='__main__':
    if not any(x=='--config' or x.startswith('--config=') for x in sys.argv[1:]):
        sys.argv.extend(['--config',str(ROOT/'head-rl-score-incremental.json')])
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',type=Path,default=ROOT/'head-rl-score-incremental.json')
    known,_=parser.parse_known_args()
    runner.HeadPolicy=IncrementalFrozenTeacherPolicy
    runner.PPO=PointPairwiseTrainer
    runner.development_gate=incremental_gate
    runner.reference_advantages=intervention_advantages
    runner.save_checkpoint=pairwise_checkpoint
    runner.write=protocol_write
    runner.main()
    config=json.loads(known.config.read_text())
    print('incremental_selection '+json.dumps(select_incremental(ROOT/config['out'])),flush=True)
