import copy
import unittest
from jevmahjong.endgame.context import annotate_record, trailing_context_active, TRAILING_INPUT_VERSION
from jevmahjong.endgame.reference import inference_request
from desktop.jev_sidecar.unified_policy import serving_request


def position():
    return {'state': {'seat': 1, 'dealer': 3, 'round_wind': 0, 'hand_number': 4,
        'honba': 0, 'sticks': 0, 'scores': [32000, 22000, 25000, 21000],
        'hand': '2m 3m 4p 5p 6p 6s 7s 8s 2p 2p', 'dora_indicators': '9m',
        'riichi': [], 'rivers': ['', '', '', ''],
        'melds': [[], [{'type': 1, 'tiles': '5z 5z 5z'}], [], []],
        'last_discard': {'seat': 2, 'tile': '4m'}},
        '_meta': {'id': 'example', 'future_rank': 1},
        'questions': {'action': {'type': 'choice', 'criteria': {
            'win': {'type': 'hora'}, 'pass': {'type': 'none'}},
            'target': {'win': 0., 'pass': 1.}, 'label': 'pass', 'src': 'endgame_terminal'}}}


class ScopedInputTests(unittest.TestCase):
    def test_current_public_rank_and_late_window_determine_scope(self):
        state = position()['state']
        self.assertTrue(trailing_context_active(state))
        state['hand_number'] = 3
        self.assertFalse(trailing_context_active(state))
        state['round_wind'] = 1
        self.assertTrue(trailing_context_active(state))
        state['scores'] = [25000] * 4
        self.assertFalse(trailing_context_active(state))
        state['seat'] = 2
        self.assertTrue(trailing_context_active(state))

    def test_active_format_is_exact_v3_with_same_targets(self):
        record = position()
        self.assertEqual(annotate_record(record, TRAILING_INPUT_VERSION),
                         annotate_record(record, 'endgame-public-v3'))
        record['state']['scores'] = [32000, 21000, 25000, 22000]
        self.assertEqual(annotate_record(record, TRAILING_INPUT_VERSION),
                         annotate_record(record, 'endgame-public-v3'))

    def test_inactive_format_is_exact_original_not_enriched_label_state(self):
        for changes in ({'hand_number': 2}, {'scores': [22000, 32000, 25000, 21000]}):
            original = position(); original['state'].update(changes)
            enriched = copy.deepcopy(original)
            enriched['state']['screen_observation'] = {'reaction': {'seat': 2, 'tile': '4m'}}
            before = copy.deepcopy(enriched)
            result = annotate_record(enriched, TRAILING_INPUT_VERSION, original_request=original)
            self.assertEqual(inference_request(result), serving_request(original))
            self.assertEqual(result['questions']['action']['target'], original['questions']['action']['target'])
            self.assertEqual(enriched, before)
            self.assertNotIn('endgame_context', result['state'])

    def test_future_outcomes_cannot_enable_scope_or_change_model_payload(self):
        record = position(); record['state']['hand_number'] = 2
        first = inference_request(annotate_record(record, TRAILING_INPUT_VERSION))
        record['_meta'].update(future_rank=4, future_scores=[0, 100000, 0, 0])
        record['questions']['action'].update(target={'win': 1., 'pass': 0.}, label='win')
        self.assertEqual(first, inference_request(annotate_record(record, TRAILING_INPUT_VERSION)))

    def test_strict_admission_is_preserved_even_outside_active_scope(self):
        record = position(); record['state'].update(hand_number=2, scores=[25000] * 3)
        with self.assertRaises(ValueError):
            annotate_record(record, TRAILING_INPUT_VERSION)


if __name__ == '__main__':
    unittest.main()
