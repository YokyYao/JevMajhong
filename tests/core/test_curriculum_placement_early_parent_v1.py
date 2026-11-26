"""The early route preserves real evidence while reporting intermediate grades."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import curriculum_placement_early_parent_v1 as parent
from tests.core.test_curriculum_placement_retained_seal_v1 import phases


def lineage():
    return dict(curriculum=dict(run_id='jev-curriculum-yaku-early-v2',
        stage='stage3B-yaku-early-v2',stage_offset=2796,
        out='runs/curriculum-yaku-early-v2/stage3B-fit'),
        training_config=dict(parent_initialization=dict(method=parent.early.METHOD,selected_step=1800)))


def phase_fixture():
    cap,queue,captured,_,receipts=phases()
    old=str(parent.retained.ENTRY.resolve());new=str(parent.ENTRY.resolve())
    for proof in cap['scheduler_proofs'].values():proof['raw']=proof['raw'].replace(old,new)
    captured['raw']=captured['raw'].replace(old,new)
    receipts={path.with_name(path.name.replace('curriculum-yaku-parent-exit-',parent.PREFIX)):value
        for path,value in receipts.items()}
    return cap,queue,captured,receipts


def completion_fixture(passed):
    cap,queue,_,_=phase_fixture()
    cap.update(status='early_direct3B_actual_terminals_preserved',
        automatic_release=False,final_confirmation_opened=False)
    done=dict(status='yaku2500_fit_and_real_curves_complete',optimizer_steps=2500,
        actual_records_seen=182494,loss_method=parent.retained.PARENT_LOSS)
    evidence=dict(status='yaku2500_final_evidence_audit_passed',optimizer_steps=2500,
        actual_records_seen=182494,loss_method=parent.retained.PARENT_LOSS,
        saved_final_reloaded_menus=24516,all376_optimizer_states_match=True,
        source_and_data_hashes_unchanged=True,source_split_replay_and_exposure_exact=True,
        final_reserved_grid_unused=True,automatic_release=False,
        stage3B_advancement=dict(passed=passed,development_only_not_independent_strength=True))
    return lineage(),done,evidence,cap,queue


class EarlyPlacementParentTests(unittest.TestCase):
    def test_intermediate_failed_grade_is_preserved_without_inventing_completion(self):
        values=completion_fixture(False);before=copy.deepcopy(values)
        with patch.object(parent,'require_registration') as registered:
            parent.require_completion(*values);registered.assert_called_once_with(values[0])
        self.assertEqual(values,before);self.assertFalse(values[2]['stage3B_advancement']['passed'])
        for key,value in (('optimizer_steps',2499),('actual_records_seen',182493)):
            bad=copy.deepcopy(values);bad[1][key]=value
            with patch.object(parent,'require_registration'),self.assertRaises(ValueError):
                parent.require_completion(*bad)

    def test_incomplete_audit_or_wrong_actual_phase_is_still_refused(self):
        for field,value in (('saved_final_reloaded_menus',24515),('all376_optimizer_states_match',False),
                ('source_and_data_hashes_unchanged',False),('final_reserved_grid_unused',False)):
            values=completion_fixture(False);values[2][field]=value
            with self.subTest(field=field),patch.object(parent,'require_registration'),self.assertRaises(ValueError):
                parent.require_completion(*values)
        values=completion_fixture(False);values[3]['scheduler_proofs']['fit']['successful']=False
        with patch.object(parent,'require_registration'),self.assertRaises(ValueError):parent.require_completion(*values)

    def test_actual_1800_parent_lineage_is_required(self):
        parent.require_lineage(lineage())
        for value in (2472,2400):
            bad=lineage();bad['training_config']['parent_initialization']['selected_step']=value
            with self.assertRaises(ValueError):parent.require_lineage(bad)
        bad=lineage();bad['curriculum']['stage_offset']=3468
        with self.assertRaises(ValueError):parent.require_lineage(bad)

    def test_all_four_phase_identities_and_allocated_gpu_terminals_are_required(self):
        cap,q,captured,receipts=phase_fixture()
        exits,paths=parent.phase_proofs(cap,q,captured,receipts.__getitem__)
        self.assertEqual(set(exits),{'prepare','fit','audit','capture'});self.assertEqual(len(paths),4)
        for mutation in ('other_command','live','cpu_only','wrong_capture','missing_prepare'):
            cap,q,captured,receipts=phase_fixture()
            if mutation=='other_command':captured['raw']=captured['raw'].replace(str(parent.ENTRY.resolve()),'/other')
            elif mutation=='live':captured['raw']=captured['raw'].replace('COMPLETED','RUNNING')
            elif mutation=='cpu_only':captured['raw']=captured['raw'].replace('gres/gpu=1','gres/gpu=0')
            elif mutation=='wrong_capture':q['capture_job']='999'
            else:del cap['scheduler_proofs']['prepare']
            with self.subTest(mutation=mutation),self.assertRaises((ValueError,RuntimeError,KeyError)):
                parent.phase_proofs(cap,q,captured,receipts.__getitem__)

    def test_only_the_explicit_report_only_seal_can_enter(self):
        sealed=dict(placement_parent_lineage_method=parent.METHOD,selected3A_parent_step=1800,
            placement_parent_selected_step=2500,stage3B_grade_is_report_not_admission_veto=True,
            parent_registration=str(parent.PARENT),original_parent_grade=dict(passed=False))
        parent.require_sealed_lineage(sealed)
        for key in ('stage3B_grade_is_report_not_admission_veto','placement_parent_lineage_method'):
            bad=dict(sealed);del bad[key]
            with self.assertRaises(ValueError):parent.require_sealed_lineage(bad)

    def test_missing_real_parent_cannot_be_admitted(self):
        with patch.object(parent,'resource'),patch.object(parent,'read',side_effect=FileNotFoundError):
            with self.assertRaises(FileNotFoundError):parent.parent_artifacts(parent.PARENT)

    def test_parent_loader_restores_global_on_error(self):
        previous=object();initialization=SimpleNamespace(verified_parent=previous)
        with self.assertRaises(RuntimeError):
            with parent.parent_loader(initialization):
                self.assertIs(initialization.verified_parent,parent.verify_parent)
                raise RuntimeError('fixture')
        self.assertIs(initialization.verified_parent,previous)


if __name__=='__main__':unittest.main()
