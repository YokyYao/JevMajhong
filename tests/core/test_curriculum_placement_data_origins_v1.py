"""Protect supervision, candidate order and actual file splits in virtual copies."""
import copy
import hashlib
import unittest

from tools import prove_curriculum_placement_complete_data_origins_v1 as subject


class PlacementDataOriginTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(state=dict(hand='1m 2m'),questions=dict(action=dict(src='source',
            criteria=dict(first=dict(type='discard'),second=dict(type='reach')),
            target=dict(first=.3,second=.7),label='second')),
            _meta=dict(id='original',group_id='family',split='validation'))
        self.catalogue={'original':dict(split='train',sha256=subject.row_sha(self.row),
            group='family',source='source')}

    def copy_row(self,row=None):
        result=copy.deepcopy(row or self.row); original=result['_meta']['id']
        result['_meta']['id']='placement-repeat-v1:'+hashlib.sha256(original.encode()).hexdigest()+':1'
        result['_meta'][subject.REPEAT_FIELD]=dict(original_id=original,copy_index=1,
            total_exposures_per_pass=2,adds_independent_data=False)
        return result

    def test_exact_copy_restores_parent_without_reinterpreting_legacy_meta_split(self):
        replica=self.copy_row(); before=copy.deepcopy(replica)
        restored,rid,index=subject.restore_copy(replica,{'source':2})
        self.assertEqual((rid,index),('original',1))
        subject.require_origin(restored,'train',self.catalogue)
        self.assertEqual(replica,before)

    def test_changed_target_cannot_hide_behind_a_valid_repeat_trace(self):
        replica=self.copy_row(); replica['questions']['action']['target']=dict(first=.9,second=.1)
        restored,_,_=subject.restore_copy(replica,{'source':2})
        with self.assertRaises(ValueError): subject.require_origin(restored,'train',self.catalogue)

    def test_copy_of_validation_cannot_become_training_even_if_legacy_tag_says_train(self):
        row=copy.deepcopy(self.row); row['_meta']['id']='heldout'; row['_meta']['split']='train'
        catalogue={'heldout':dict(split='validation',sha256=subject.row_sha(row),group='family',source='source')}
        restored,_,_=subject.restore_copy(self.copy_row(row),{'source':2})
        with self.assertRaises(ValueError): subject.require_origin(restored,'train',catalogue)

    def test_changed_candidate_order_is_not_object_exact_model_input(self):
        changed=copy.deepcopy(self.row)
        changed['questions']['action']['criteria']=dict(reversed(list(changed['questions']['action']['criteria'].items())))
        self.assertEqual(changed,self.row)
        with self.assertRaises(ValueError): subject.require_origin(changed,'train',self.catalogue)


if __name__=='__main__': unittest.main()
