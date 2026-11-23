import copy
import unittest
from tools.curriculum_rank_nonwin_teacher_v2 import candidate_keys,soft_target


def outcome(rank):return dict(full_match_completed=True,rank=rank,top2=rank<=2,net_from_anchor=0)


class FullRiichiProposalTests(unittest.TestCase):
    def test_riichi_compares_all_discards_even_outside_ordinary_cap(self):
        criteria={f'd{i}':dict(type='dahai',sh=i%3) for i in range(14)}
        criteria['ri']=dict(type='reach')
        request=dict(state=dict(seat=0,riichi=[]),questions=dict(action=dict(criteria=criteria)))
        original=copy.deepcopy(request);scores={k:1/15 for k in criteria}
        got=candidate_keys(request,scores,'d3')
        self.assertEqual(got[0],'d3');self.assertEqual(len(got),15)
        self.assertEqual(set(got),set(criteria));self.assertEqual(request,original)
        reordered=copy.deepcopy(request)
        reordered['questions']['action']['criteria']=dict(reversed(list(criteria.items())))
        self.assertEqual(set(candidate_keys(reordered,scores,'d3')),set(criteria))

    def test_ordinary_candidate_budget_remains_four(self):
        criteria={f'd{i}':dict(type='dahai',sh=i%3) for i in range(14)}
        request=dict(state=dict(seat=0,riichi=[]),questions=dict(action=dict(criteria=criteria)))
        self.assertEqual(len(candidate_keys(request,{k:1/14 for k in criteria},'d3')),4)

    def test_full_menu_soft_targets_include_every_compared_arm(self):
        keys=[f'd{i}' for i in range(14)]+['ri']
        worlds=[{k:outcome(2 if k=='ri' else 4) for k in keys} for _ in range(32)]
        got=soft_target(worlds,keys,4,1123)
        self.assertEqual(set(got['target']),set(keys))
        self.assertTrue(all(p>0 for p in got['target'].values()))
        self.assertAlmostEqual(sum(got['target'].values()),1.)
        self.assertEqual(got['label'],'ri')
        self.assertGreater(got['target']['ri'],.98)
        with self.assertRaises(ValueError):soft_target(worlds[:31],keys,4,1123)

    def test_duplicate_or_invalid_riichi_menu_is_rejected(self):
        request=dict(state=dict(seat=0,riichi=[]),questions=dict(action=dict(criteria={
            'd':dict(type='dahai'),'r1':dict(type='reach'),'r2':dict(type='reach')})))
        with self.assertRaises(ValueError):candidate_keys(request,{'d':.4,'r1':.3,'r2':.3},'d')

