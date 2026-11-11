import unittest
import copy
import json
import torch
from pathlib import Path
import tempfile
from tools.run_experiment import freeze_source
from jevmahjong.training.sampling import stratum, weights
from jevmahjong.data.native_pairs import ENCODER_FILES, load_sources
from jevmahjong.core.io import digest, request_hash


def row(kind, delta):
    return {'category': kind, 'action': 1, 'teacher_action': 0, 'fast_advantage': delta,
        'public_request': {'questions': {'action': {'criteria': {'a0': {'type': 'none'}, 'a1': {'type': kind}}}}}}


def native_collection(root, start, gamma=.995):
    root.mkdir(); (root / 'walls-private').mkdir()
    config = {'game_mode': '4p-red-east', 'seed_scheme': 'spaced-v1', 'fast_gamma': gamma,
        'training_seed_start': start, 'training_seeds_per_stage': 4, 'training_replicas': 1,
        'intervention_categories': ['calling']}
    files = {role: {name: 'fixed-' + name for name in ENCODER_FILES} for role in ('discard', 'call')}
    (root / 'protocol.json').write_text(json.dumps({'config': config, 'model_files': files}))
    for seed in range(start, start + 4):
        events = [{'type': 'start_kyoku', 'scores': [25000] * 4, 'kyotaku': 0}, {'type': 'tsumo'},
            {'type': 'ryukyoku', 'deltas': [0] * 4}, {'type': 'end_kyoku'}, {'type': 'end_game'}]
        controls = [dict(seed=seed, seat=seat, replica=-1, score=25000, mjai_log=events,
            point_components={'draw_settlement': 0}, all_point_components=[{'draw_settlement': 0}] * 4,
            settlement_ledger_passed=True) for seat in range(4)]
        games = [dict(g, replica=0, trainable_decisions=int(g['seat'] == 0),
            reused_control_outcome=g['seat'] != 0) for g in controls]
        trace = dict(row('chi', 0.), category='calling', scope='calling', eligible=[0, 1],
            alternative_probability=1., episode_key=(seed, 0, 0), hand_index=0, draws_before=0,
            control_fast_reward=0., branch_fast_reward=0.)
        trace['request_sha256'] = request_hash(trace['public_request'])
        p = root / 'walls-private' / f'{seed}.pt'
        torch.save({'seed': seed, 'controls': controls, 'games': games, 'rows': [trace]}, p)
        p.with_suffix('.audit.json').write_text(json.dumps({'sha256': digest(p)}))
    (root / 'collection-result.json').write_text(json.dumps({'status': 'collection_complete', 'walls': 4, 'rows': 4}))
    return files


class SamplingTests(unittest.TestCase):
    def test_combined_collections_keep_all_seats_and_branches_in_whole_wall_split(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            files = native_collection(a, 100); native_collection(b, 104)
            partitions, hashes, sources = load_sources([a, b], files)
            self.assertEqual([r['episode_key'][0] for r in partitions['held']], [103, 107])
            self.assertEqual(len(partitions['train']), 6)
            self.assertEqual(len(hashes), 8)
            self.assertEqual(sum(s['train_walls'] for s in sources), 6)
            utility = {'mode': 'risk_adjusted_fast_win_v1', 'dealin_penalty': .3, 'net_weight': .1, 'point_scale': 8000.}
            relabeled, _, _ = load_sources([a, b], files, utility)
            old, new = partitions['train'][0], relabeled['train'][0]
            self.assertEqual(old['public_request'], new['public_request'])
            self.assertEqual(old['request_sha256'], new['request_sha256'])
            self.assertNotIn('training_advantage', old)
            self.assertEqual(new['training_advantage'], 0.)
            with self.assertRaisesRegex(ValueError, 'Unique'):
                load_sources([a, a])
            different = copy.deepcopy(files); different['call']['adapter_config.json'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'encoders'):
                load_sources([a], different)

    def test_duplicate_walls_reward_mismatch_and_tampered_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b, c = (Path(directory) / n for n in ('a', 'b', 'c'))
            native_collection(a, 100); native_collection(b, 100); native_collection(c, 104, .99)
            with self.assertRaisesRegex(ValueError, 'Whole wall repeated'):
                load_sources([a, b])
            with self.assertRaisesRegex(ValueError, 'compatible'):
                load_sources([a, c])
            (a / 'walls-private/100.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'shard changed'):
                load_sources(a)

    def test_complete_marker_cannot_hide_missing_registered_walls(self):
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / 'a'; native_collection(a, 100)
            (a / 'walls-private/100.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'coverage differs'):
                load_sources(a)

    def test_editing_main_code_keeps_existing_run_snapshot_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); package = root / 'jevmahjong'; package.mkdir()
            file = package / '__init__.py'; file.write_text('VERSION = 1\n')
            first = freeze_source(root)
            file.write_text('VERSION = 2\n')
            second = freeze_source(root)
            self.assertNotEqual(first, second)
            self.assertEqual((first / 'jevmahjong/__init__.py').read_text(), 'VERSION = 1\n')
            self.assertEqual((second / 'jevmahjong/__init__.py').read_text(), 'VERSION = 2\n')

    def test_known_mass_and_direction_are_balanced_without_copying_rows(self):
        rows = [row('chi', 1.) for _ in range(3)] + [row('chi', -1.)] + [row('pon', 1.), row('pon', -1.)]
        w, report = weights(rows, {'chi': .5, 'pon': .5}, 4.)
        self.assertAlmostEqual(sum(w[:4]), sum(w[4:]))
        self.assertAlmostEqual(sum(w[:3]), w[3])
        self.assertEqual(report['copies_added'], 0)
        self.assertIn('ankan/positive', report['missing_action_directions'])

    def test_skipping_beneficial_operation_reverses_label_and_rare_weight_is_capped(self):
        r = row('daiminkan', -1.); r['teacher_action'], r['action'] = 1, 0
        self.assertEqual(stratum(r), ('daiminkan', 'positive'))
        rows = [row('chi', 1.) for _ in range(30)] + [row('daiminkan', 1.)]
        w, report = weights(rows, {'chi': .5, 'daiminkan': .5}, 4.)
        self.assertEqual(max(w), 4.)
        self.assertLess(report['effective_action_mass']['daiminkan'], .5)


if __name__ == '__main__': unittest.main()
