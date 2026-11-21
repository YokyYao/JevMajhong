import copy
import unittest

from tools.curriculum_riichi_context import FIELD, dora_kind, enrich_request, tile


def request():
    return dict(state=dict(seat=0, dealer=0, round_wind=0, honba=1, sticks=2,
        scores=[23000, 26000, 25000, 26000], hand='1z 1z 1z 2m 3m 4m 5m 6m 7m 2p 3p 4p 0s 6s',
        melds=[[], [], [], []], rivers=['1m 9m', '5s', '', ''], riichi=[1], dora_indicators='4s'),
        questions=dict(action=dict(type='choice', instructions='Choose a legal action.', label='a0', src='loss-only',
            criteria=dict(a0=dict(type='reach'), a1=dict(type='dahai', pai='5sr', sh=0, ukeire=3, safe=[1]),
                          a2=dict(type='dahai', pai='6s', sh=0, ukeire=2, safe=[])))))


class RiichiContextTests(unittest.TestCase):
    def test_red_and_honor_notation_and_dora_wrap(self):
        self.assertEqual(tile('0s'), tile('5sr'))
        self.assertNotEqual(tile('5s'), tile('5sr'))
        self.assertEqual(tile('E'), tile('1z'))
        self.assertEqual([dora_kind(k) for k in (8, 30, 33)], [0, 27, 31])

    def test_no_label_metadata_leak_and_training_inference_agreement(self):
        original = request()
        altered = copy.deepcopy(original)
        altered['questions']['action'].update(label='a2', src='another-loss-tag', target={'a2': 1.})
        altered['_meta'] = dict(future_reward=99999)
        a = enrich_request(original)
        b = enrich_request(altered)
        self.assertEqual(a['questions']['action']['criteria'], b['questions']['action']['criteria'])
        public = dict(state=original['state'], questions=dict(action={k: v for k, v in original['questions']['action'].items()
            if k in ('type', 'instructions', 'criteria')}))
        self.assertEqual(enrich_request(public)['questions']['action']['criteria'], a['questions']['action']['criteria'])
        self.assertNotIn(FIELD, original['questions']['action']['criteria']['a0'])

    def test_summaries_preserve_red_discard_and_double_wind_roles(self):
        result = enrich_request(request())
        c = result['questions']['action']['criteria']['a0'][FIELD]
        self.assertEqual(c['value_honor_roles_min'], 2)
        self.assertEqual((c['red_after_discard_min'], c['red_after_discard_max']), (0, 1))
        self.assertEqual((c['dora_after_discard_min'], c['dora_after_discard_max']), (0, 1))
        self.assertEqual(c['safe_tenpai_discard_options'], 1)
        self.assertEqual(c['safe_tenpai_ukeire_max'], 3)
        self.assertEqual(c['gap_to_second'], 3000)
        self.assertEqual(enrich_request(result), result)

    def test_no_riichi_window_is_unchanged(self):
        value = request()
        del value['questions']['action']['criteria']['a0']
        self.assertEqual(enrich_request(value), value)

    def test_unknown_dora_is_not_imputed_and_stale_context_is_rejected(self):
        value = request()
        value['state']['dora_indicators'] = None
        result = enrich_request(value)
        self.assertIsNone(result['questions']['action']['criteria']['a0'][FIELD]['dora_after_discard_max'])
        result['questions']['action']['criteria']['a0'][FIELD]['self_score'] = -1
        with self.assertRaises(ValueError):
            enrich_request(result)
