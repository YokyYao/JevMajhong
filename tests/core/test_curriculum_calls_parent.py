import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.curriculum_calls_parent import qualify, scheduler_snapshot
from tools.curriculum_parent import ParentNotReady


class CallsParentTests(unittest.TestCase):
    def setUp(self):
        self.reg = dict(registered_final_step=324, train_records=20237, curriculum=dict(
            stage='stage2-calls-v1', candidate_rule='registered_final_full_two_epochs_324_updates',
            epochs=2, batch=1, accumulation=125, validation_every_steps=81,
            development_targets=dict(max_plain_agreement_drop_from_step0=.02,
                max_riichi_balanced_accuracy_drop_from_step0=.05,
                min_five_call_type_balanced_accuracy_gain_from_step0=.03,
                min_final64_net_gain_over_parent=0., max_final64_dealin_increase_over_parent=.02)))
        self.done = dict(status='stage2_fit_and_curves_complete', optimizer_steps=324, actual_records_seen=40474)
        self.curve = [dict(step=i, stratum_accuracy={'plain': .5}, riichi_type_counts=dict(tp=1, fn=3, tn=3, fp=1),
            operation_type_counts={op: dict(tp=1 if i == 0 else 2, fn=3 if i == 0 else 2, tn=3, fp=1)
                for op in ('chi', 'pon', 'ankan', 'daiminkan', 'kakan')}) for i in (0, 81, 162, 243, 324)]
        self.summary = dict(matches=64, seed_groups=8, net_points=100.)
        self.audit = dict(status='stage2_final_evidence_audit_passed', optimizer_steps=324, real_records_seen=40474,
            source_and_data_hashes_unchanged=True, heldout_groups_excluded=True, exact_train_and_validation_replay=True,
            all376_optimizer_states_match=True, saved_final_reloaded_menus=4721, saved_final_max_probability_error=1e-7,
            checked_complete_games=224, comparison_to_parent=dict(net_points={'gain': 10.}, dealin={'gain': 0.}),
            final_summary=self.summary.copy(), stage2_advancement=dict(plain_agreement_gain=0., riichi_balanced_gain=0.,
                five_call_type_balanced_gain=.125, passed=True, development_only_not_strength_proof=True))

    def test_complete_final_can_qualify(self):
        self.assertTrue(qualify(self.reg, self.done, self.audit, self.curve, self.summary)['passed'])

    def test_flag_does_not_hide_actual_call_collapse_or_net_loss(self):
        bad = copy.deepcopy(self.curve)
        bad[-1]['operation_type_counts']['chi'] = dict(tp=0, fn=4, tn=0, fp=4)
        with self.assertRaises(ParentNotReady):
            qualify(self.reg, self.done, self.audit, bad, self.summary)
        bad = copy.deepcopy(self.audit)
        bad['comparison_to_parent']['net_points']['gain'] = -1.
        with self.assertRaises(ParentNotReady):
            qualify(self.reg, self.done, bad, self.curve, self.summary)

    def test_partial_exposure_missing_reload_and_small_grid_fail(self):
        with self.assertRaises(ParentNotReady):
            qualify(self.reg, dict(self.done, optimizer_steps=81), self.audit, self.curve, self.summary)
        for changes in (dict(saved_final_reloaded_menus=100), dict(saved_final_max_probability_error=float('nan')),
                        dict(heldout_groups_excluded=False)):
            with self.assertRaises(ParentNotReady):
                qualify(self.reg, self.done, dict(self.audit, **changes), self.curve, self.summary)
        with self.assertRaises(ParentNotReady):
            qualify(self.reg, self.done, self.audit, self.curve, dict(self.summary, matches=16))
        context = copy.deepcopy(self.reg)
        context['curriculum']['riichi_input_schema'] = 'public-riichi-context-v1'
        with self.assertRaises(ParentNotReady):
            qualify(context, self.done, self.audit, self.curve, self.summary)

    def test_only_captured_completed_scheduler_state_can_replace_expired_history(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory)/'observations.jsonl'
            known = dict(job_id=12, state='RUNNING', exit_code='0:0', source='current_scheduler_query')
            file.write_text(json.dumps(dict(utc='observed', jobs={'12': known}))+'\n')
            missing = subprocess.CompletedProcess([], 1, '', 'Invalid job id specified')
            with patch('tools.curriculum_calls_parent.subprocess.run', return_value=missing):
                with self.assertRaises(ParentNotReady):
                    scheduler_snapshot(dict(job_id=12, exit_code=0), file)
                known['state'] = 'COMPLETED'
                file.write_text(json.dumps(dict(utc='observed', jobs={'12': known}))+'\n')
                result = scheduler_snapshot(dict(job_id=12, exit_code=0), file)
                self.assertEqual(result['source'], 'preserved_verified_completed_scheduler_observation')
                self.assertEqual(result['snapshot']['state'], 'COMPLETED')
                with self.assertRaises(ParentNotReady):
                    scheduler_snapshot(dict(job_id=12, exit_code=1), file)
            # A transient query failure or a currently RUNNING job must never
            # be replaced by a stale successful snapshot.
            for response in (subprocess.CompletedProcess([], 1, '', 'Controller timeout'),
                             subprocess.CompletedProcess([], 0, 'JobId=12 JobState=RUNNING ExitCode=0:0', '')):
                with patch('tools.curriculum_calls_parent.subprocess.run', return_value=response):
                    with self.assertRaises(ParentNotReady):
                        scheduler_snapshot(dict(job_id=12, exit_code=0), file)


if __name__ == '__main__':
    unittest.main()
