import copy
import unittest

from tools.curriculum_win_context import project_trigger, with_win_context, public_tile, SCHEMA


class WinSourceTests(unittest.TestCase):
    def setUp(self):
        self.request = dict(state=dict(seat=2, rivers=['', '1m', '', '9p'],
            last_discard=32), questions=dict(action=dict(type='choice',
            instructions='fixture', criteria=dict(a0=dict(type='hora'), a1=dict(type='none')))))

    def test_ron_uses_public_current_actor_and_preserves_original_menu(self):
        original = copy.deepcopy(self.request)
        got = with_win_context(self.request, {'a0':dict(kind='ron', tile='1m')},
            project_trigger(dict(type='dahai', actor=1, pai='1m', secret='discarded'), 2))
        self.assertEqual(self.request, original)
        self.assertEqual(got['questions'], original['questions'])
        self.assertEqual(got['state']['win_context'], dict(schema=SCHEMA,
            candidates={'a0':dict(kind='ron', source='discard', source_seat=1, tile='1m')}))

    def test_robbed_added_kan_never_uses_stale_discard_source(self):
        got = with_win_context(self.request, {'a0':dict(kind='ron', tile='3s')},
            project_trigger(dict(type='kakan', actor=0, pai='3s'), 2))
        self.assertEqual(got['state']['win_context']['candidates']['a0']['source_seat'], 0)
        self.assertEqual(got['state']['win_context']['candidates']['a0']['source'], 'robbed_kakan')

    def test_self_draw_and_opponent_draw_redaction(self):
        got = with_win_context(self.request, {'a0':dict(kind='tsumo', tile='5p')},
            project_trigger(dict(type='tsumo', actor=2, pai='5p'), 2))
        self.assertEqual(got['state']['win_context']['candidates']['a0']['source'], 'self_draw')
        self.assertEqual(project_trigger(dict(type='tsumo', actor=0, pai='8s', hands=[1]), 2),
            dict(type='tsumo', actor=0))

    def test_mismatched_unknown_or_self_ron_rejected(self):
        for event in (dict(type='dahai', actor=1, pai='2m'), dict(type='dahai', actor=2, pai='1m'),
                      dict(type='tsumo', actor=1, pai='1m'), dict(type='dahai', actor=3, pai='1m')):
            with self.assertRaises(ValueError):
                with_win_context(self.request, {'a0':dict(kind='ron', tile='1m')}, project_trigger(event, 2))
        for event in (dict(type='hora', actor=1, pai='1m'), dict(type='dahai', actor=True, pai='1m'),
                      dict(type='dahai', actor=1, pai='?')):
            with self.assertRaises(ValueError): project_trigger(event, 2)
        with self.assertRaises(ValueError):
            with_win_context(self.request, {'a1':dict(kind='ron', tile='1m')},
                project_trigger(dict(type='dahai', actor=1, pai='1m'), 2))

    def test_ankan_identity_is_public_and_strict(self):
        self.assertEqual(project_trigger(dict(type='ankan', actor=1, consumed=['0p','5p','5p','5p']), 2),
            dict(type='ankan', actor=1, tile='5p'))
        with self.assertRaises(ValueError):
            project_trigger(dict(type='ankan', actor=1, consumed=['1p','2p','1p','1p']), 2)

    def test_native_honor_and_red_notation_matches_model_tiles(self):
        self.assertEqual(public_tile('E'), '1z')
        self.assertEqual(public_tile('5pr'), '0p')
        self.assertEqual(public_tile('0p'), '0p')
        for bad in ('0z', '8z', '?', '', None, '10m'):
            with self.assertRaises(ValueError): public_tile(bad)


if __name__ == '__main__': unittest.main()
