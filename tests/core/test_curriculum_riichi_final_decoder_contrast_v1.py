import copy
import unittest
from tools.diagnostics.riichi_final_decoder_contrast_v1 import require_final,contrast
from tools.curriculum_basic_repair import MASS


class FinalDecoderContrast(unittest.TestCase):
    def test_incomplete_integrity_cannot_be_used_as_final(self):
        done=dict(status='stage1_fit_and_curves_complete',optimizer_steps=608,
            actual_records_seen=57140,curve_steps=[0,152,304,456,608])
        evidence=dict(status='stage1_final_evidence_audit_passed',optimizer_steps=608,
            real_records_seen=57140,source_and_data_hashes_unchanged=True,
            final_summary=dict(matches=64,seed_groups=8),basic_repair_source_quality={'all_source_quality_passed':False})
        require_final(done,evidence)
        for field,value in [('optimizer_steps',532),('actual_records_seen',49999),('curve_steps',[0,152,304,456])]:
            bad=dict(done,**{field:value})
            with self.subTest(field=field),self.assertRaises(ValueError):require_final(bad,evidence)
        for key,value in [('status','pending'),('source_and_data_hashes_unchanged',False),('final_summary',{'matches':16,'seed_groups':2})]:
            with self.subTest(field=key),self.assertRaises(ValueError):require_final(done,dict(evidence,**{key:value}))

    def fixture(self,discards):
        rows=[];predictions=[]
        for source in MASS:
            plain=source.endswith(':plain');yes=source.endswith(':riichi_yes')
            criteria={str(i):{'type':'dahai'} for i in range(discards)}
            if not plain:criteria['r']={'type':'reach'}
            label='r' if yes else '0'
            rows.append(dict(_meta=dict(id=source,group_id=source),questions={'action':dict(src='curriculum1:'+source,criteria=criteria,label=label)}))
            p=[1/discards]*discards if plain else [.6/discards]*discards+[.4]
            predictions.append(dict(id=source,keys=list(criteria),p=p))
        return rows,predictions

    def test_dispersion_contrast_keeps_raw_predictions_and_fixed_rule(self):
        rows,predictions=self.fixture(10);before=copy.deepcopy((rows,predictions))
        result=contrast(rows,predictions)
        self.assertEqual((rows,predictions),before)
        self.assertEqual(result['pooled']['confusion']['raw_argmax'],dict(tp=2,fn=0,tn=0,fp=2))
        self.assertEqual(result['pooled']['confusion']['fixed_full_class_0_5'],dict(tp=0,fn=2,tn=2,fp=0))
        self.assertFalse(result['actual_policy_decisions_changed'])
        self.assertFalse(result['threshold_search'])

    def test_one_fixed_contrast_does_not_assert_it_is_better(self):
        rows,predictions=self.fixture(10)
        for p in predictions:
            if not p['id'].endswith(':plain'):p['p']=[.02]*10+[.8]
        result=contrast(rows,predictions)
        self.assertEqual(result['pooled']['confusion']['raw_argmax'],result['pooled']['confusion']['fixed_full_class_0_5'])
        self.assertEqual(result['pooled']['fixed_decoder_disagreements'],{})


if __name__=='__main__':unittest.main()
