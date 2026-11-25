import copy
import dis
import unittest

from tools import curriculum_yaku_ri_anchor_profile_v1 as profile


class AugmentedYakuProfileTests(unittest.TestCase):
    def fixture(self):
        return ([{'_meta':{'id':name}} for name in ('old','new','route')],
            [{'id':name} for name in ('old','new','route')],
            dict(original=['old'],supplement=['new'],route=['route']),
            dict(original=1,supplement=1,route=1))

    def test_disjoint_complete_cohorts_and_predictions(self):
        groups,rows,saved=profile.partition(*self.fixture())
        self.assertEqual(set(rows),set(saved));self.assertEqual(set(rows),set.union(*groups.values()))
        for which in ('records','predictions'):
            values=copy.deepcopy(self.fixture());index=0 if which=='records' else 1
            values[index][1]=copy.deepcopy(values[index][0])
            with self.subTest(which=which),self.assertRaises(ValueError):profile.partition(*values)

    def test_overlap_missing_extra_and_duplicate_members_refused(self):
        for change in ('overlap','missing','extra','duplicate'):
            values=copy.deepcopy(self.fixture())
            if change=='overlap':values[2]['supplement']=['old']
            elif change=='missing':values[2].pop('route')
            elif change=='extra':values[2]['unknown']=[]
            else:values[2]['route']=['route','route']
            with self.subTest(change=change),self.assertRaises(ValueError):profile.partition(*values)

    def test_only_cardinality_constants_and_replay_scope_change(self):
        actual=profile.augmented_function();original=profile.original.profile
        self.assertEqual(actual.__code__.co_code,original.__code__.co_code)
        self.assertEqual(actual.__code__.co_names,original.__code__.co_names)
        for old,new in [(24516,25826),(23688,24998)]:
            self.assertEqual(sum(i.opname=='LOAD_CONST' and i.argval==old for i in dis.get_instructions(original)),1)
            self.assertEqual(sum(i.opname=='LOAD_CONST' and i.argval==new for i in dis.get_instructions(actual)),1)
        self.assertIs(actual.__globals__['replay_profile'],profile.inherited.profile)
        self.assertIsNot(original.__globals__['replay_profile'],profile.inherited.profile)

    def test_exact_tensor_and_full_parent_parity_required_before_updates(self):
        proof=dict(exact_tensors=dict(parent_adapter_exact=True,parent_head_exact=True,adapter_tensors=372,head_tensors=4),
            all_parent_prediction_parity=dict(parent_menus=24998,child_menus=25826,choices_equal=True,max_probability_error=0.),
            checked_before_first_optimizer_update=True,step0_bundle_identity='child',parent_bundle_identity='parent')
        identity={'identity_sha256':'child'};reg={'curriculum':{'parent_identity':'parent'}}
        profile.require_step0(identity,reg,proof)
        for key,value in [('parent_menus',23688),('choices_equal',False),('max_probability_error',5e-5)]:
            bad=copy.deepcopy(proof);bad['all_parent_prediction_parity'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):profile.require_step0(identity,reg,bad)
        bad=copy.deepcopy(proof);bad['exact_tensors']['head_tensors']=3
        with self.assertRaises(ValueError):profile.require_step0(identity,reg,bad)


if __name__=='__main__':unittest.main()
