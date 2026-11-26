from types import SimpleNamespace
import unittest
from riichienv import Phase
from tools.curriculum_native_decisions import decision_scope,decision_inventory


class NativeDecisionScopeTests(unittest.TestCase):
    def test_draw_and_post_call_have_distinct_scopes(self):
        source=SimpleNamespace(active_players=[2],phase=Phase.WaitAct);obs=SimpleNamespace(player_id=2)
        self.assertEqual(decision_scope(source,obs,dict(type='tsumo',actor=2)),'draw')
        for kind in ('chi','pon'):self.assertEqual(decision_scope(source,obs,dict(type=kind,actor=2)),'post_call')

    def test_only_current_discard_response_is_admitted(self):
        source=SimpleNamespace(active_players=[1,2],phase=Phase.WaitResponse);obs=SimpleNamespace(player_id=2)
        self.assertEqual(decision_scope(source,obs,dict(type='dahai',actor=3)),'response')
        for kind in ('kakan','ankan'):
            with self.assertRaises(ValueError):decision_scope(source,obs,dict(type=kind,actor=3))

    def test_inactive_or_unspecified_scope_fails_closed(self):
        with self.assertRaises(ValueError):decision_scope(SimpleNamespace(active_players=[1],phase=Phase.WaitAct),SimpleNamespace(player_id=2),dict(type='tsumo',actor=2))
        with self.assertRaises(ValueError):decision_inventory(None,'unverified')


if __name__=='__main__':unittest.main()
