import time
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from desktop.jev_sidecar.engine import Board, candidates, parse_melds, request, self_kan_candidates, hand_strategy, reaction_candidates
from desktop.jev_sidecar.autoplay import plan_click
from desktop.jev_sidecar.screen_state import ScreenStateReader
from desktop.jev_sidecar.tiles import parse
from desktop.jev_sidecar.vision import Detection


class KanRegressionTests(unittest.TestCase):
    def test_open_kan_consumes_three_owned_tiles_and_uses_call_button(self):
        event = {'tile': '6z', 'seat': 1, 'offered': ['daiminkan']}
        values = reaction_candidates(Board(parse('123m123p123s1666z'), reaction=event))
        kan = next(v for v in values if v['action']['type'] == 'daiminkan')
        self.assertEqual(kan['action']['consumed'], ['F'] * 3)
        screen = {'reaction': event, 'offered': ['daiminkan']}
        self.assertIsNotNone(plan_click({'top3': [kan], 'all_candidates': values}, [], screen, (), (1, 1)))

    def test_live_four_green_dragons_reaches_model_and_clicker(self):
        hand = parse('6m3578p40688s6666z')
        reader = ScreenStateReader()
        reader.counter_reader.read = Mock(return_value={})
        detector = Mock()
        detector.detect.return_value = []
        found = lambda h: {'hand': [Detection(t, .99, (0, 0, 10, 20)) for t in h]}
        image = Image.new('RGB', (400, 200))
        with patch.object(reader, 'buttons', return_value=['daiminkan']):
            screen = reader.read(image, {}, detector, found(hand))
        self.assertTrue(screen['self_kan_available'])
        self.assertIsNone(screen['reaction'])
        board = Board(hand, self_kan_available=screen['self_kan_available'])
        values = candidates(board)
        kan = next(v for v in values if v.get('action', {}).get('type') == 'ankan')
        self.assertEqual(kan['action']['consumed'], ['F'] * 4)
        self.assertIn(kan['action']['type'], [v['type'] for v in request(board, values)['questions']['action']['criteria'].values()])
        result = {'top3': [kan], 'all_candidates': values}
        self.assertEqual(plan_click(result, found(hand)['hand'], screen, (0, 0, 1, 1), (400, 200)).label, '暗杠 发')
        # Supplementary draw removes four owned tiles, adds one and preserves
        # the concealed-kan type even when its hidden faces cannot be detected.
        reader.expected_self_kan = {'time': time.monotonic(), 'hand': hand, 'melds': [], 'action': kan['action']}
        remaining = [t for t in hand if t != '6z'] + ['1m']
        with patch.object(reader, 'buttons', return_value=[]):
            after = reader.read(image, {}, detector, found(remaining))
        self.assertEqual(after['melds'], parse_melds('a:6666z'))
        self.assertTrue(candidates(Board(remaining, melds=[after['melds'], [], [], []])))

    def test_kan_requires_game_offer_and_preserves_owned_red(self):
        board = Board(parse('0555m123p123s1122z'))
        self.assertEqual(self_kan_candidates(board), [])
        board.self_kan_available = True
        action = self_kan_candidates(board)[0]['action']
        self.assertEqual(action['consumed'], ['5m', '5m', '5m', '5mr'])
        # A kan available in a different tile does not authorize a post-riichi
        # kan of a concealed block unrelated to the drawn tile.
        board.riichi, board.drawn = [0], '2z'
        self.assertEqual(self_kan_candidates(board), [])

    def test_added_kan_and_multiple_choices(self):
        board = Board(parse('123m123p123s56z'), melds=[parse_melds('555z'), [], [], []], self_kan_available=True)
        kan = self_kan_candidates(board)[0]
        self.assertEqual(kan['action'], {'type': 'kakan', 'pai': 'P', 'consumed': ['P'] * 3})
        screen = {'self_kan_available': True, 'offered': ['daiminkan']}
        self.assertIsNotNone(plan_click({'top3': [kan], 'all_candidates': [kan]}, [], screen, (), (1, 1)))
        self.assertIsNone(plan_click({'top3': [kan], 'all_candidates': [kan, kan]}, [], screen, (), (1, 1)))

    def test_unopened_hand_does_not_imply_refusing_calls(self):
        board = Board(parse('11m3589p233440s'))
        text = hand_strategy(board, {'action': {'type': 'none'}}, direction_only=True)
        self.assertNotIn('保留门清', text)
        self.assertIn('副露', text)

    def test_added_kan_transition_does_not_create_an_extra_meld(self):
        reader = ScreenStateReader()
        reader.counter_reader.read = Mock(return_value={})
        reader.melds = parse_melds('555z')
        hand = parse('123m123p123s56z')
        action = self_kan_candidates(Board(hand, melds=[reader.melds, [], [], []], self_kan_available=True))[0]['action']
        reader.previous_hand = hand
        reader.expected_self_kan = {'time': time.monotonic(), 'hand': hand,
                                   'melds': reader.melds, 'action': action}
        after = [t for t in hand if t != '5z'] + ['7z']
        detector = Mock()
        detector.detect.return_value = []
        found = {'hand': [Detection(t, .99, (0, 0, 10, 20)) for t in after]}
        with patch.object(reader, 'buttons', return_value=[]):
            screen = reader.read(Image.new('RGB', (400, 200)), {}, detector, found)
        self.assertEqual(screen['melds'], parse_melds('k:5555z'))
        self.assertIsNone(reader.expected_self_kan)


if __name__ == '__main__':
    unittest.main()
