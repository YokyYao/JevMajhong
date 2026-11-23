import unittest

from tools.prepare_curriculum_placement_exposures_v1 import proposed_schedule


class PlacementExposurePreparationTests(unittest.TestCase):
    def test_full_two_epoch_budget_avoids_tiny_tail_and_pre_save_evaluation(self):
        for total in range(91247, 100001, 137):
            schedule = proposed_schedule(total)
            self.assertEqual(schedule['expected_record_exposures'], 2*total)
            self.assertEqual(schedule['proposed_final_step'], 2*schedule['updates_per_epoch'])
            self.assertGreaterEqual(schedule['final_update_records'], .75*schedule['accumulation'])
            points = schedule['validation_and_game_steps']
            self.assertEqual(len(points), 5)
            self.assertEqual(points, sorted(set(points)))
            self.assertEqual(points[-1], schedule['proposed_final_step'])
            self.assertEqual([i for i in range(0, points[-1], schedule['eval_every_steps'])], points[:-1])
            self.assertTrue(all(step % schedule['checkpoint_every_steps'] == 0 for step in points[:-1]))
            self.assertFalse(schedule['registered_training_budget'])
