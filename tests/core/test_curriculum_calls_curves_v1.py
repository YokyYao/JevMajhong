import copy
import csv
import io
import tempfile
from pathlib import Path
import unittest

from tools import export_curriculum_calls_curves_v1 as curves


class CallsCurveTests(unittest.TestCase):
    def train(self, step):
        counts = {k: 1 for k in curves.MASS}; counts[next(iter(counts))] = 241
        objective = {k: 1. for k in counts}; ce = {k: 1.2 for k in counts}
        return dict(local_step=step, global_step=608+step, stage=curves.STAGE, loss_method=curves.METHOD,
            loss=1., ce=1.2, lr=.00001, head_lr_derived=.00005, records_seen=step*253,
            reward=None, source_counts=counts, source_objective=objective, source_ce=ce,
            grad_norm=123., private_case='must never be published')

    def game(self, step=0, net=800.):
        return dict(local_step=step, global_step=608+step, stage=curves.STAGE,
            reward_definition='terminal_raw_net_points_div8000_v1', matches=16, seed_groups=2,
            hand_win_rate=.2, dealin_hand_rate=.1, first_rate=.25, top2_rate=.5,
            fourth_rate=.25, net_points=net, reward=net/8000)

    def test_every_actual_update_and_original_CE_are_required(self):
        rows = [self.train(1), self.train(2)]; weights = {k: 1. for k in curves.MASS}
        curves.check_train(rows, 2, weights)
        for changed in (rows[1:], rows[:1], [rows[0], rows[0]]):
            with self.assertRaises(ValueError): curves.check_train(changed, 2, weights)
        for name, value in [('loss_method', 'hard_ce'), ('ce', 1.), ('reward', 0.),
                            ('records_seen', 1), ('loss', float('nan')), ('global_step', 1)]:
            changed = copy.deepcopy(rows); changed[-1][name] = value
            with self.assertRaises(ValueError): curves.check_train(changed, 2, weights)

    def test_live_snapshot_ignores_only_an_incomplete_last_append(self):
        self.assertEqual(curves.complete_jsonl(b'{"x":1}\n{"x":'), [{'x': 1}])
        with self.assertRaises(ValueError): curves.complete_jsonl(b'{"x":1}\ninvalid\n')
        for rows in ([{'local_step': 194}], [{'local_step': 0}, {'local_step': 388}], [{'local_step': False}]):
            with self.assertRaises(ValueError): curves.prefix(rows, 388)

    def test_public_train_CSV_excludes_gradients_private_diagnostics_and_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'train.csv'; curves.csv_file(path, curves.TRAIN_FIELDS, [self.train(1)])
            text = path.read_text(); parsed = list(csv.DictReader(io.StringIO(text)))
            self.assertEqual(tuple(parsed[0]), curves.TRAIN_FIELDS)
            for private in ('grad_norm', 'private_case', 'source_counts', 'source_objective', 'reward'):
                self.assertNotIn(private, text)

    def test_matched16_final64_remain_distinct_real_grids(self):
        rows = [self.game(), self.game(194)]
        final = self.game(388, 4000.); final.update(matches=64, seed_groups=8)
        final['curve_grid_subset'] = {k: v for k, v in self.game(388, -800.).items() if k in curves.GAME_FIELDS}
        rows.append(final)
        matched, full = curves.check_games(rows, 388, 388)
        self.assertEqual(matched[-1]['net_points'], -800.)
        self.assertEqual(full[-1]['net_points'], 4000.)
        bad = copy.deepcopy(rows); bad[-1].pop('curve_grid_subset')
        with self.assertRaises(ValueError): curves.check_games(bad, 388, 388)
        bad = copy.deepcopy(rows); bad[0]['reward'] = 1.
        with self.assertRaises(ValueError): curves.check_games(bad, 388, 388)

    def test_final_publication_needs_complete_independent_evidence_not_a_quality_waiver(self):
        evidence = dict(status='stage2_calls_classroute_final_evidence_audit_passed', optimizer_steps=388,
            real_records_seen=98044, saved_final_reloaded_menus=12479, saved_final_max_probability_error=0.,
            checked_complete_games=224, all376_optimizer_states_match=True, actual_parent16_full_events_exact=True,
            source_and_data_hashes_unchanged=True, exact_train_validation_replay=True,
            stage2_advancement={'passed': False})
        points = [{'local_step': s} for s in (0, 194, 388)]
        curves.require_final_packet(evidence, 388, points, points, [{}])
        for field, value in [('optimizer_steps', 387), ('saved_final_reloaded_menus', 1),
                             ('actual_parent16_full_events_exact', False), ('checked_complete_games', 96)]:
            with self.assertRaises(ValueError): curves.require_final_packet(dict(evidence, **{field: value}), 388, points, points, [{}])


if __name__ == '__main__': unittest.main()
