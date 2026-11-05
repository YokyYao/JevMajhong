import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
from desktop.jev_sidecar.engine import Adviser, Board, candidates, parse_melds, request, shanten
from desktop.jev_sidecar.tiles import dora, index, parse
from desktop.jev_sidecar.vision import Detection, Templates, nms


class TileTests(unittest.TestCase):
    def test_compact_red_honors_and_multiplicity(self):
        self.assertEqual(parse('123m 05p 東白中'), ['1m', '2m', '3m', '0p', '5p', '1z', '5z', '7z'])
        with self.assertRaises(ValueError):
            parse('55550m')
        with self.assertRaises(ValueError):
            parse('8z')
        self.assertEqual([dora(t) for t in ['9m', '4z', '7z']], [0, 27, 31])

    def test_tenpai_discard_and_visible_ukeire(self):
        board = Board(parse('123456m345p67s115z'))
        result = Adviser().recommend(board)
        self.assertEqual(result['top3'][0]['tile'], '5z')
        self.assertEqual(result['top3'][0]['sh'], 0)
        self.assertEqual(result['top3'][0]['ukeire'], 8)
        board.rivers[1] = parse('555s')
        self.assertEqual(Adviser().recommend(board)['top3'][0]['ukeire'], 5)

    def test_complete_hand_is_marked_before_discard_advice(self):
        board = Board(parse('123456m345p678s11z'))
        self.assertTrue(Adviser().recommend(board)['winning_shape'])

    def test_riichi_lock_and_unknown_public_information(self):
        board = Board(parse('123456m345p67s115z'), riichi=[0], drawn='5z')
        self.assertEqual([v['tile'] for v in candidates(board)], ['5z'])
        board.drawn = None
        with self.assertRaises(ValueError):
            candidates(board)
        board.drawn = '5z'
        self.assertIsNone(request(board, candidates(board))['state']['rivers'][1])

    def test_invalid_frame_and_visible_counts_rejected(self):
        with self.assertRaises(ValueError):
            candidates(Board(parse('123456m345p67s11z')))
        board = Board(parse('123456m345p67s115z'), indicators=parse('111z'))
        with self.assertRaises(ValueError):
            candidates(board)

    def test_open_hand_and_human_readable_melds(self):
        melds = parse_melds('555m,a:1111z')
        self.assertEqual([m['type'] for m in melds], [1, 3])
        with self.assertRaises(ValueError):
            parse_melds('124m')
        board = Board(parse('123m456p67s115z'), melds=[parse_melds('555m'), [], [], []])
        self.assertEqual(Adviser().recommend(board)['top3'][0]['tile'], '5z')

    def test_native_feature_parity_on_real_observation(self):
        try:
            from riichienv import RiichiEnv
            from semantic import build_request, describe
        except ImportError:
            self.skipTest('Training environment reference engine is optional on Windows')
        obs = RiichiEnv(game_mode='4p-red-east', seed=42).reset()[0]
        state, _, _ = describe(obs)
        board = Board(parse(state['hand']), rivers=[parse(r) for r in state['rivers']],
                      indicators=parse(state['dora_indicators']), riichi=state['riichi'])
        original, _, _ = build_request(obs)
        expected = {json.dumps({'pai': v['pai'], 'type': v['type']}, sort_keys=True): v
                    for v in original['questions']['action']['criteria'].values() if v['type'] == 'dahai'}
        actual = request(board, candidates(board))['questions']['action']['criteria']
        for value in actual.values():
            key = json.dumps({'pai': value['pai'], 'type': value['type']}, sort_keys=True)
            self.assertEqual(value, expected[key])

    def test_shanten_matches_training_engine_random_hands(self):
        try:
            from riichienv import calculate_shanten
        except ImportError:
            self.skipTest('Training reference engine is optional')
        generator = random.Random(66)
        for _ in range(100):
            types = tuple(sorted(t // 4 for t in generator.sample(range(136), 13)))
            self.assertEqual(shanten(types), calculate_shanten([i * 4 for i in types]))


class VisionTests(unittest.TestCase):
    def test_overlap_suppression_keeps_separate_same_tiles(self):
        a = Detection('1m', .99, (0, 0, 30, 50))
        b = Detection('2m', .8, (1, 1, 29, 49))
        c = Detection('1m', .9, (35, 0, 65, 50))
        self.assertEqual(nms([a, b, c]), [a, c])

    def test_user_teaching_requires_aligned_tiles_and_recovers_exact_faces(self):
        with tempfile.TemporaryDirectory() as temporary:
            bank = Templates(temporary)
            image = Image.new('RGB', (100, 80), 'white')
            draw = ImageDraw.Draw(image)
            draw.rectangle((8, 8, 30, 68), fill='black')
            draw.ellipse((60, 15, 90, 60), fill='green')
            boxes = [(0, 0, 48, 80), (52, 0, 100, 80)]
            with self.assertRaises(ValueError):
                bank.teach(image, boxes, ['1m'])
            bank.teach(image, boxes, ['1m', '2s'])
            self.assertEqual(bank.classify(image.crop(boxes[0]))[0], '1m')
            self.assertEqual(bank.classify(image.crop(boxes[1]))[0], '2s')


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication
        from desktop.jev_sidecar.app import STYLE
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setStyleSheet(STYLE)

    def test_demo_and_stale_inference_result(self):
        from desktop.jev_sidecar.app import Sidecar, Task
        with tempfile.TemporaryDirectory() as temporary, patch('desktop.jev_sidecar.app.CONFIG', Path(temporary) / 'settings.json'):
            panel = Sidecar(demo=True)
            panel.show()
            deadline = time.monotonic() + 5
            while panel.jobs and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            self.assertIn('白', panel.hero.text())
            self.app.processEvents()
            self.assertEqual(panel.tabs.currentWidget().horizontalScrollBar().maximum(), 0,
                             'A wide screenshot must not force horizontal scrolling in the narrow sidebar')
            before = panel.hero.text()
            panel.generation += 1
            task = Task('analysis', lambda: {}, panel.generation - 1)
            panel.finished({'task': task, 'kind': 'analysis', 'generation': panel.generation - 1, 'result': {}})
            self.assertEqual(panel.hero.text(), before)
            panel.timer.stop()
            panel.close()


if __name__ == '__main__':
    unittest.main()
