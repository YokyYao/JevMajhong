"""Reuse completed on-policy training features; change only the return baseline adapter."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from head_rl import HeadPolicy, fingerprint, mixed_probabilities, reference_advantages, write
from reference_calibration import calibrated_baselines
from scripts import train_head_rl as runner

original_write = write


class CachedTrainingPolicy(HeadPolicy):
    def __init__(self, config, checkpoint=None):
        if checkpoint is not None:
            raise ValueError('Cached experiment starts from its frozen behavior checkpoint, not a resume')
        self.cache_source = ROOT / config['cached_training_source']
        source = json.loads((self.cache_source / 'protocol.json').read_text())
        for key in ['reference', 'exploration_uniform', 'game_mode', 'reward_mode', 'reward_scale',
                    'seed_scheme', 'training_seed_start', 'training_seeds_per_stage', 'training_replicas']:
            if config[key] != source['config'][key]:
                raise ValueError(f'Cached behavior differs: {key}')
        for name, expected in source['code_sha256'].items():
            if fingerprint(ROOT / name) != expected:
                raise ValueError(f'Cache source numerical code changed: {name}')
        reference = ROOT / config['reference']
        for name, field in [('head.pt', 'reference_head_sha256'),
                            ('adapter_model.safetensors', 'reference_adapter_sha256')]:
            if fingerprint(reference / name) != source[field]:
                raise ValueError('Cached behavior weights changed')
        super().__init__(config)
        games = json.loads((self.cache_source / 'training-games-stage-01.json').read_text())
        controls = json.loads((self.cache_source / 'reference-controls-stage-01.json').read_text())
        self.cached_games = {(g['seed'], g['seat'], g['replica']): g for g in games}
        self.cached_controls = {(g['seed'], g['seat']): g for g in controls}
        rows = torch.load(self.cache_source / 'rollout-stage-01.pt', map_location='cpu', weights_only=False)
        self.cached_rows = {key: [] for key in self.cached_games}
        for row in rows:
            self.cached_rows[tuple(row['episode_key'])].append(row)
        # The stored behavior probabilities must match this exact frozen pointer head.
        with torch.no_grad():
            for row in rows:
                logits = self.head(row['decide'].to('cuda'), row['options'].to('cuda'))
                probability = mixed_probabilities(logits, row['eligible'], config['exploration_uniform'])
                if not torch.allclose(probability.cpu(), row['reference'], atol=2e-5, rtol=2e-4):
                    raise ValueError('Cached features differ from the frozen behavior probabilities')
                if abs(float(probability[row['action']].log()) - row['old_logp']) > 2e-4:
                    raise ValueError('Cached sampled-action likelihood differs')
        original_write(ROOT / config['out'] / 'cache-verification.json', {
            'source_run': str(self.cache_source), 'source_protocol_sha256': source['sha256'],
            'source_rollout_sha256': fingerprint(self.cache_source / 'rollout-stage-01.pt'),
            'source_head_sha256': source['reference_head_sha256'],
            'all_cached_feature_probabilities_verified': len(rows),
            'reused_training_games': len(games), 'reused_controls': len(controls),
            'new_simulated_training_games': 0})

    def game(self, seed, seat, training=False, frozen=False, old_opponent=False, replica=0):
        if training:
            key = (seed, seat, replica)
            game = self.cached_games[key]
            if game['old_opponent'] != old_opponent:
                raise ValueError('Cached opponent composition differs')
            return dict(game), [dict(row) for row in self.cached_rows[key]]
        if frozen and replica == -1:
            game = self.cached_controls[seed, seat]
            if game['old_opponent'] != old_opponent:
                raise ValueError('Cached control opponents differ')
            return dict(game), []
        return super().game(seed, seat, training, frozen, old_opponent, replica)


def calibrated_advantages(controls, games, rows, replicas, scale):
    diagnostic = reference_advantages(controls, games, rows, replicas, scale)
    baselines, fit = calibrated_baselines(games, controls)
    for row in rows:
        row['uncalibrated_control_return'] = row['baseline_return']
        row['baseline_return'] = baselines[tuple(row['episode_key'])]
        row['advantage'] = (row['return'] - row['baseline_return']) / scale
    diagnostic['baseline_calibration'] = fit
    return diagnostic


def protocol_write(path, data):
    if Path(path).name == 'protocol.json':
        data = copy.deepcopy(data)
        data['code_sha256'].update({name: fingerprint(ROOT / name) for name in
                                   ['reference_calibration.py', 'scripts/train_calibrated_cache.py']})
        data['training_data_mode'] = 'reused frozen on-policy feature cache, no new training games'
    original_write(path, data)


if __name__ == '__main__':
    runner.HeadPolicy = CachedTrainingPolicy
    runner.reference_advantages = calibrated_advantages
    runner.write = protocol_write
    runner.main()
