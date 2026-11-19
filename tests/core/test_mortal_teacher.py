import copy
import json
import unittest

from jevmahjong.teachers.mortal import public_events, legal_index
from tools.build_mortal_teacher import bucket, smoothed_target, select_rows, assert_split


class DummyAction:
    def __init__(self, value): self.value=value
    def to_mjai(self): return json.dumps(self.value)


class TeacherPrivacyTests(unittest.TestCase):
    def test_own_tiles_are_retained_but_hidden_draw_is_rejected(self):
        events=[{'type':'tsumo','actor':0,'pai':'5mr'}]
        self.assertEqual(public_events(events,0),events)
        with self.assertRaises(ValueError): public_events(events,1)

    def test_opponent_initial_hand_is_rejected_without_modifying_input(self):
        event={'type':'start_kyoku','tehais':[['?']*13 for _ in range(4)]}
        event['tehais'][0][0]='1m';original=copy.deepcopy(event)
        with self.assertRaises(ValueError): public_events([event],1)
        self.assertEqual(event,original)
        self.assertEqual(public_events([event],0),[event])

    def test_red_tiles_and_consumed_tiles_are_not_collapsed(self):
        a=DummyAction({'type':'pon','actor':0,'pai':'5m','consumed':['5mr','5m']})
        b=DummyAction({'type':'pon','actor':0,'pai':'5m','consumed':['5m','5m']})
        translated=DummyAction({'type':'pon','actor':0,'pai':'5m','consumed':['5m','5mr']})
        self.assertEqual(legal_index(translated,[b,a]),1)
        with self.assertRaises(ValueError): legal_index(translated,[a,a])

    def test_no_rule_fallback_for_unmapped_action(self):
        with self.assertRaises(ValueError):
            legal_index(DummyAction({'type':'reach'}),[DummyAction({'type':'none'})])


class TeacherDatasetTests(unittest.TestCase):
    def test_pass_does_not_become_a_positive_kan_label(self):
        c={'a0':{'type':'daiminkan'},'a1':{'type':'none'}}
        self.assertEqual(bucket(c,'a1',[]),'teacher_kan')
        p=smoothed_target(list(c),'a1',.02)
        self.assertGreater(p['a1'],p['a0'])
        self.assertAlmostEqual(sum(p.values()),1.)

    def test_win_menu_never_distills_hanchan_teacher(self):
        self.assertIsNone(bucket({'a0':{'type':'hora'},'a1':{'type':'none'}},'a1',[]))

    def test_positive_calls_and_defense_are_separate(self):
        self.assertEqual(bucket({'a':{'type':'chi'}},'a',[]),'teacher_chi')
        self.assertEqual(bucket({'a':{'type':'dahai'}},'a',[2]),'teacher_defense')
        self.assertEqual(bucket({'a':{'type':'chi'},'b':{'type':'none'}},'b',[]),'teacher_reaction')

    def test_whole_source_games_are_held_out(self):
        row={'_meta':{'group_id':'game1','id':'a'}}
        with self.assertRaises(ValueError): assert_split([row],[{'_meta':{'group_id':'game1','id':'b'}}])

    def test_quota_shortfall_fails_and_group_cap_is_enforced(self):
        sources=['teacher_kan','teacher_chi','teacher_pon','teacher_reach','teacher_defense','teacher_reaction','teacher_discard']
        rows=[{'questions':{'action':{'src':s}},'_meta':{'id':s,'group_id':'one','selection_hash':s}} for s in sources]
        with self.assertRaises(ValueError): select_rows(rows,{s:1 for s in sources},1)
        self.assertEqual(len(select_rows(rows,{s:1 for s in sources},7)),7)


if __name__=='__main__': unittest.main()
