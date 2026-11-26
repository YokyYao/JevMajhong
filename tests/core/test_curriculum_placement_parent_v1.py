import copy
from types import SimpleNamespace
import unittest

from tools import curriculum_placement_parent_v1 as parent
from tools.curriculum_placement_mix_v1 import NEW_SOURCES


def final_fixture():
    return [dict(registered_final_step=2500),
        dict(optimizer_steps=2500,status='yaku2500_fit_and_real_curves_complete',actual_records_seen=182494),
        dict(status='yaku2500_final_evidence_audit_passed',stage3B_advancement={'passed':True},
            saved_final_reloaded_menus=24516,source_and_data_hashes_unchanged=True,
            all376_optimizer_states_match=True,final_reserved_grid_unused=True,
            final_identity='model',final_policy_identity='policy'),
        dict(status='actual_yaku2500_full_evidence_independently_audited',
            original_launch_binding_unchanged=True,actual_fit_job=12),
        dict(status='actual_final3B_fit_and_audit_terminal_evidence_preserved_not_child_admitted',
            selected_step=2500,strategy_grade_overridden=False,original_strategy_grade={'passed':True},
            parent_fit_job=12,final_identity='model',final_policy_identity='policy')]


def rows():
    return [dict(_meta=dict(id='original',group_id='family',split='train'),state={'hand':['1m']},
        questions=dict(action=dict(src='old',instructions='ordinary',label='a',
            criteria={'a':{'type':'dahai'},'b':{'type':'reach'}},target={'a':.6,'b':.4}))) ]


def project(row):
    result=copy.deepcopy(row);result['questions']['action']['instructions']='placement'
    return result


class PlacementParentTests(unittest.TestCase):
    def test_only_original_qualified_actual_final_is_eligible(self):
        parent.require_final(*final_fixture())
        for index,field,value in ((0,'registered_final_step',2499),(1,'optimizer_steps',2499),
                (1,'actual_records_seen',20),(2,'stage3B_advancement',{'passed':False}),
                (2,'final_reserved_grid_unused',False),(4,'strategy_grade_overridden',True),
                (4,'final_identity','other')):
            bad=final_fixture();bad[index][field]=value
            with self.assertRaises(ValueError):parent.require_final(*bad)

    def test_only_declared_instruction_projection_changes_replay(self):
        old=rows();before=copy.deepcopy(old);child=[project(r) for r in old]
        extra=copy.deepcopy(child[0]);extra['_meta']['id']='new';extra['questions']['action']['src']=NEW_SOURCES[0]
        proof=parent.replay_projection(old,child+[extra],project)
        self.assertEqual(old,before)
        self.assertEqual(proof['parent_records'],1)
        self.assertEqual(proof['projected_instruction_changes'],1)
        self.assertTrue(proof['whole_parent_prediction_parity_not_claimed'])

    def test_changed_state_label_menu_order_missing_duplicate_or_unknown_replay_fails(self):
        old=rows();good=[project(r) for r in old]
        bad=copy.deepcopy(good);bad[0]['state']['hand']=['2m']
        with self.assertRaises(ValueError):parent.replay_projection(old,bad,project)
        bad=copy.deepcopy(good);bad[0]['questions']['action']['label']='b'
        with self.assertRaises(ValueError):parent.replay_projection(old,bad,project)
        with self.assertRaises(ValueError):parent.replay_projection(old,[],project)
        with self.assertRaises(ValueError):parent.replay_projection(old,good+good,project)
        bad=copy.deepcopy(good);extra=copy.deepcopy(good[0]);extra['_meta']['id']='unknown'
        with self.assertRaises(ValueError):parent.replay_projection(old,bad+[extra],project)
        def reorder(row):
            out=project(row);q=out['questions']['action'];q['criteria']=dict(reversed(list(q['criteria'].items())))
            return out
        with self.assertRaises(ValueError):parent.replay_projection(old,[reorder(old[0])],reorder)

    def test_parent_loader_restores_original_even_after_failure(self):
        sentinel=object();initialization=SimpleNamespace(verified_parent=sentinel)
        with self.assertRaises(RuntimeError):
            with parent.parent_loader(initialization):
                self.assertIs(initialization.verified_parent,parent.verify_parent)
                raise RuntimeError('synthetic failure')
        self.assertIs(initialization.verified_parent,sentinel)
