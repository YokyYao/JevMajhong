import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tools.prepare_curriculum_placement_launch_v1 import require_integration,require_control_state,extend,terminal


def proof():
    return dict(status='actual_placement_upstream_two_update_integration_passed',optimizer_steps=2,
        actual_control_record_exposures=114,control_sources=57,complete_prepared_epoch_orders_exercised=2,
        job_id='319427',node='ws-test-001',parent_identity='qualified-parent',
        actual_updates_proven=True,parent_tensors_unchanged=True,isolated_from_student_fit=True,
        actual_weighted_loss_journal_matches=True,saved_optimizer_states_verified=True,
        fixtures_from_current_TRAIN_only=True,control_validation_not_original_VAL=True,
        control_distribution_is_not_full_course_mix=True,control_weights_are_not_a_candidate=True,
        fit_launch_admitted=False,automatic_release=False,
        exact_inheritance=dict(parent_adapter_exact=True,parent_head_exact=True,adapter_tensors=372,head_tensors=4))


class PlacementLaunchTests(unittest.TestCase):
    def test_wrong_parent_job_scope_or_fake_control_admission_is_rejected(self):
        reg=dict(parent_model=dict(identity_sha256='qualified-parent'));require_integration(proof(),reg,319427)
        for field,value in (('parent_identity','other'),('job_id','old'),('optimizer_steps',1),
            ('actual_control_record_exposures',2),('control_weights_are_not_a_candidate',False),
            ('fit_launch_admitted',True),('node','login-01')):
            bad=proof();bad[field]=value
            with self.assertRaises(ValueError):require_integration(bad,reg,319427)

    def test_full_optimizer_state_is_required_instead_of_only_summary_flags(self):
        state=dict(optimizer_step=2,records_seen=114,args=dict(resume=''),
            optimizer=dict(state={i:dict(step=2) for i in range(376)}))
        require_control_state(state,2)
        for kind in ('old_momentum_step','missing_tensor','wrong_exposure','resume'):
            bad=copy.deepcopy(state)
            if kind=='old_momentum_step':bad['optimizer']['state'][375]['step']=2502
            elif kind=='missing_tensor':bad['optimizer']['state'].pop(375)
            elif kind=='wrong_exposure':bad['records_seen']=113
            else:bad['args']['resume']='parent'
            with self.assertRaises(ValueError):require_control_state(bad,2)

    def test_conflicting_bindings_are_not_silently_overwritten(self):
        files={'source':'old'};extend(files,{'source':'old','criteria':'fixed'})
        with self.assertRaises(ValueError):extend(files,{'source':'different'})
        self.assertEqual(files['source'],'old')

    def test_only_original_successful_gpu_terminal_can_be_reused_without_query(self):
        import json
        with TemporaryDirectory() as directory:
            script=Path(directory)/'actual.slurm';archive=Path(directory)/'terminal.json'
            receipt=dict(job_id='319427',exit_code=0)
            raw=('JobId=319427 JobState=COMPLETED ExitCode=0:0 NodeList=ws-test-001 '
                 'AllocTRES=cpu=4,gres/gpu=1 StartTime=2026-11-24T01:00:00 '
                 'EndTime=2026-11-24T01:10:00 Command='+str(script.resolve()))
            saved=dict(source='actual_terminal_scontrol',raw=raw,receipt=receipt)
            archive.write_text(json.dumps(saved))
            with patch('tools.prepare_curriculum_placement_launch_v1.subprocess.run',side_effect=AssertionError('Must reuse original archive')):
                self.assertEqual(terminal(319427,receipt,script,archive),saved)
                with self.assertRaises(ValueError):terminal(319428,receipt,script,archive)
            for change in (('COMPLETED','FAILED'),('gres/gpu=1','gres/gpu=0'),('ws-test-001','login-01')):
                bad=dict(saved,raw=raw.replace(*change));archive.write_text(json.dumps(bad))
                with self.assertRaises(Exception):terminal(319427,receipt,script,archive)
