import copy
import unittest
from tools.curriculum_basic_repair import grade
from tools.curriculum_parent import ParentNotReady
from tools.seal_curriculum_basic_repair import require_source_quality


class BasicRepairParentTests(unittest.TestCase):
    def fixture(self):
        reg={'curriculum':{'source_specific_parent_quality_required':True},'registered_final_step':608}
        counts={s+':'+k:10 for s in ('legacy','teacher') for k in ('plain','riichi_yes','riichi_no')}
        validation={'local_step':608,'source_stratum_counts':counts,'stratum_counts':{'plain':20,'riichi_yes':20,'riichi_no':20},
            'riichi_by_source':{s:{'tp':8,'fn':2,'tn':8,'fp':2} for s in ('legacy','teacher')},
            'riichi_type_counts':{'tp':16,'fn':4,'tn':16,'fp':4},'stratum_accuracy':{'plain':.8},'stratum_ce':{'plain':.5},
            'source_stratum_accuracy':{'legacy:plain':.8},'source_stratum_ce':{'legacy:plain':.5}}
        quality=grade(validation)
        evidence={'status':'stage1_final_evidence_audit_passed','source_specific_parent_seal_required':True,
            'basic_repair_source_quality':quality,'stage1_advancement':quality['original_aggregate_gate']}
        return reg,evidence,validation

    def test_actual_separate_passes_are_required_not_only_an_aggregate_header(self):
        reg,evidence,validation=self.fixture()
        self.assertTrue(require_source_quality(reg,evidence,validation)['all_source_quality_passed'])
        del evidence['basic_repair_source_quality']
        with self.assertRaises(ParentNotReady):require_source_quality(reg,evidence,validation)

    def test_forged_success_cannot_hide_an_actual_failed_legacy_class(self):
        reg,evidence,validation=self.fixture()
        validation['riichi_by_source']['legacy'].update(tn=4,fp=6)
        validation['riichi_type_counts'].update(tn=12,fp=8)
        with self.assertRaises(ParentNotReady):require_source_quality(reg,evidence,validation)
        calculated=grade(validation);self.assertTrue(calculated['original_aggregate_gate']['passed'])
        evidence.update(basic_repair_source_quality=calculated,stage1_advancement=calculated['original_aggregate_gate'])
        with self.assertRaises(ParentNotReady):require_source_quality(reg,evidence,validation)

    def test_missing_or_inconsistent_actual_source_counts_are_rejected(self):
        for change in ('missing','confusion'):
            reg,evidence,validation=self.fixture()
            if change=='missing':del validation['source_stratum_counts']['teacher:plain']
            else:validation['riichi_by_source']['teacher']['tp']+=1
            with self.assertRaises(ParentNotReady):require_source_quality(reg,evidence,validation)

    def test_incomplete_or_wrong_registered_stage_cannot_be_sealed(self):
        reg,evidence,validation=self.fixture();changed=copy.deepcopy(validation);changed['local_step']=76
        with self.assertRaises(ParentNotReady):require_source_quality(reg,evidence,changed)
        reg['curriculum']['source_specific_parent_quality_required']=False
        with self.assertRaises(ParentNotReady):require_source_quality(reg,evidence,validation)


if __name__=='__main__':unittest.main()
