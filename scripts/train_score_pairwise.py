"""Controlled loss change on the exact completed single-decision training cache."""
import copy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from head_rl import fingerprint, save_checkpoint, write
from point_pairwise import PointPairwiseTrainer
from scripts.train_calibrated_cache import CachedTrainingPolicy
from scripts.train_single_decision import intervention_advantages
from scripts.audit_score_confirmation import audit_rows
from scripts import train_head_rl as runner


class ControlledCachePolicy(CachedTrainingPolicy):
    def __init__(self, config, checkpoint=None):
        super().__init__(config, checkpoint)
        previous = json.loads((self.cache_source/'protocol.json').read_text())['config']
        if any(previous[k] != config[k] for k in ['development_seed_start', 'development_seeds']):
            raise ValueError('Controlled loss comparison requires the identical frozen development baseline')
        source = self.cache_source/'development-baseline/games.jsonl'
        rows = [json.loads(s) for s in source.read_text().splitlines()]
        audit_config = {**config, 'confirmation_seed_start': config['development_seed_start'],
                        'confirmation_seeds': config['development_seeds']}
        audit_rows(rows, audit_config)
        directory = ROOT/config['out']/'development-baseline'
        directory.mkdir()
        shutil.copyfile(source, directory/'games.jsonl')
        write(ROOT/config['out']/'baseline-reuse-verification.json', {
            'source_run': str(self.cache_source), 'source_baseline_sha256': fingerprint(source),
            'reused_native_greedy_baseline_matches': len(rows), 'new_baseline_matches': 0,
            'same_frozen_head_code_seed_coverage_and_opponents': True})


def pairwise_checkpoint(*args, **kwargs):
    path = save_checkpoint(*args, **kwargs)
    # Keep portable checkpoint metadata truthful without changing historical runtime code.
    payload = torch.load(path/'head.pt', map_location='cpu', weights_only=False)
    payload['training_method'] = 'point-delta-weighted reference-relative action ranking'
    payload['ppo_surrogate_used'] = False
    temporary = path/'head.pt.tmp'
    torch.save(payload, temporary)
    temporary.replace(path/'head.pt')
    complete = json.loads((path/'complete.json').read_text())
    complete['head_sha256'] = fingerprint(path/'head.pt')
    write(path/'complete.json', complete)
    return path


def protocol_write(path, data):
    if Path(path).name == 'protocol.json':
        data = copy.deepcopy(data)
        data['code_sha256'].update({name: fingerprint(ROOT/name) for name in
            ['point_pairwise.py', 'scripts/train_score_pairwise.py', 'scripts/train_calibrated_cache.py',
             'scripts/train_single_decision.py', 'single_decision_rl.py', 'reference_calibration.py',
             'scripts/audit_score_confirmation.py', 'semantic.py', 'seed_control.py']})
        data['training_objective'] = 'point_delta_weighted_reference_relative_pairwise_ranking'
        data['ppo_surrogate_used'] = False
        data['training_data_mode'] = 'same frozen single-decision cache; no new simulated training matches'
    write(path, data)


if __name__ == '__main__':
    runner.HeadPolicy = ControlledCachePolicy
    runner.PPO = PointPairwiseTrainer
    runner.reference_advantages = intervention_advantages
    runner.save_checkpoint = pairwise_checkpoint
    runner.write = protocol_write
    runner.main()
