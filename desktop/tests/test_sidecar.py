import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
from desktop.jev_sidecar.engine import Adviser, Board, candidates, parse_melds, reaction_candidates, request, shanten
from desktop.jev_sidecar.tiles import dora, index, name, parse
from desktop.jev_sidecar.vision import Detection, Templates, TileDetector, nms, red_five


class TileTests(unittest.TestCase):
    def test_riichi_flow_keeps_chosen_discard_when_grey_tile_is_missing(self):
        from desktop.jev_sidecar.autoplay import RiichiFlow
        flow = RiichiFlow()
        hand = parse('46m12305667799p3z')
        flow.begin(hand, '3z')
        visible = [Detection(t, 1, (i * 20, 0, i * 20 + 18, 30)) for i, t in enumerate(hand) if t != '0p']
        screen = {'riichi_selecting': True}
        plan = flow.plan(visible, screen, (0, .7, 1, .3), (300, 100))
        self.assertIsNotNone(plan)
        self.assertEqual(plan.label, '打 3z')
        flow.clicked = True
        self.assertIsNone(flow.plan(visible, screen, (0, .7, 1, .3), (300, 100)))
        flow.plan(visible, {'own_riichi': True}, (0, .7, 1, .3), (300, 100))
        self.assertIsNotNone(flow.hand, 'Missing a different tile must not confirm the discard')
        after = [Detection(t, 1, (0, 0, 18, 30)) for t in hand if t != '3z']
        flow.plan(after, {'own_riichi': True}, (0, .7, 1, .3), (300, 100))
        self.assertIsNone(flow.hand)

    def test_riichi_selection_does_not_complete_when_unrelated_tile_disappears(self):
        from desktop.jev_sidecar.screen_state import ScreenStateReader
        reader = ScreenStateReader()
        hand = parse('46m12305667799p3z')
        reader.expected_riichi_hand, reader.expected_riichi_tile = hand, '3z'
        image = Image.new('RGB', (200, 100))
        def read(tiles):
            return reader.read(image, {}, None, {'hand': [Detection(t, 1, (0, 0, 1, 1)) for t in tiles]})
        with patch.object(reader, 'buttons', return_value=[]), patch('desktop.jev_sidecar.screen_state.riichi_selection_overlay', return_value=True):
            read(hand)
            state = read([t for t in hand if t != '0p'])
            self.assertTrue(state['riichi_selecting'])
            self.assertFalse(state['own_riichi'])
        with patch.object(reader, 'buttons', return_value=[]), patch('desktop.jev_sidecar.screen_state.riichi_selection_overlay', return_value=False):
            state = read([t for t in hand if t != '0p'])
            self.assertFalse(state['own_riichi'])
            state = read([t for t in hand if t != '3z'])
            self.assertTrue(state['own_riichi'])
            self.assertFalse(state['riichi_selecting'])

    def test_model_readiness_requires_successful_warmup(self):
        from unittest.mock import Mock
        adviser = Adviser()
        adviser.load = Mock(return_value='Jev CUDA')
        adviser.recommend = Mock(side_effect=RuntimeError('warmup failed'))
        phases = []
        with self.assertRaisesRegex(RuntimeError, 'warmup failed'):
            adviser.prepare('checkpoint', phases.append)
        self.assertFalse(adviser.ready)
        adviser.recommend = Mock(return_value={})
        self.assertIn('已就绪', adviser.prepare('checkpoint', phases.append))
        self.assertTrue(adviser.ready)
        adviser.recommend.assert_called_once()
        self.assertTrue(adviser.recommend.call_args.kwargs['use_model'])
        self.assertIn('预热', phases[-1])

    def test_yellow_turn_countdown_is_not_riichi_cancel(self):
        from PIL import ImageDraw
        from desktop.jev_sidecar.screen_state import riichi_selection_overlay
        image = Image.new('RGB', (1280, 741), (35, 65, 110))
        draw = ImageDraw.Draw(image)
        # Ranked turn countdown occupies the right side of the old crop.
        draw.rectangle((999, 535, 1022, 602), fill=(255, 210, 40))
        self.assertFalse(riichi_selection_overlay(image))

    def test_new_deal_clears_riichi_after_partial_settlement_frame(self):
        from desktop.jev_sidecar.screen_state import ScreenStateReader
        reader = ScreenStateReader()
        image = Image.new('RGB', (200, 100))
        from unittest.mock import Mock
        detector = Mock()
        detector.detect.return_value = []
        def read(tiles):
            return reader.read(image, {}, detector, {'hand': [Detection(t, 1, (0, 0, 1, 1)) for t in tiles]})
        with patch.object(reader, 'buttons', return_value=[]):
            read(parse('3356m345888p222s'))
            reader.own_riichi = True
            read(parse('335m345p'))  # Settlement overlay hides the rest.
            state = read(parse('2468m1457p37s247z'))
        self.assertFalse(state['own_riichi'])

    def test_auto_click_plan_matches_tile_and_executes_once_per_decision(self):
        from desktop.jev_sidecar.autoplay import AutoPlayer, plan_click
        from unittest.mock import Mock
        ds = [Detection('1m', 1, (0, 0, 20, 30)), Detection('5p', 1, (30, 0, 50, 30))]
        result = {'top3': [{'tile': '5p', 'name': '五筒'}]}
        self.assertIsNone(plan_click(result, ds, {}, (0, 0, 1, 1), (100, 100)))
        ds += [Detection('2s', 1, (60, 0, 80, 30))] * 12
        plan = plan_click(result, ds, {}, (.1, .7, .8, .2), (100, 100))
        self.assertAlmostEqual(plan.x, .5)
        self.assertAlmostEqual(plan.y, .82)
        clicker = Mock()
        player = AutoPlayer(clicker)
        self.assertTrue(player.execute('decision1', plan, None))
        self.assertFalse(player.execute('decision1', plan, None))
        self.assertEqual(clicker.call_count, 1)
        player.last_click = 0
        key = (1, ('1m', '5p'), '{}', 'discard')
        self.assertTrue(player.execute(key, plan, None))
        self.assertIn('等待游戏响应', player.observe(('1m', '5p'), '{}'))
        self.assertIsNotNone(player.pending)
        self.assertIn('画面已变化', player.observe(('1m',), '{}'))
        self.assertIsNone(player.pending)
        self.assertFalse(player.execute(key, plan, None))
        reach = {'top3': [{'action': {'type': 'reach'}, 'name': '立直', 'tile': '5p'}]}
        self.assertIsNone(plan_click(reach, ds, {}, (.1, .7, .8, .2), (100, 100)))
        self.assertIsNotNone(plan_click(reach, ds, {'riichi_available': True}, (.1, .7, .8, .2), (100, 100)))

    def test_combined_chi_pon_buttons_require_both_legal_actions(self):
        from desktop.jev_sidecar.screen_state import ScreenStateReader
        reader = ScreenStateReader()
        hand = parse('5567m2478p789s66z')
        found = {'hand': [Detection(t, 1, (0, 0, 1, 1)) for t in hand]}
        with patch.object(reader, 'buttons', return_value=['chi', 'pon']):
            result = reader.read(Image.new('RGB', (200, 100)), {}, None, found)
        self.assertEqual(result['reaction']['tile'], '5m')
        self.assertEqual(result['reaction']['seat'], 3)

    def test_strategy_describes_direction_and_actual_tenpai_waits(self):
        result = Adviser().recommend(Board(parse('123456m345p67s115z')))
        self.assertIn('立直', result['strategy'])
        self.assertIn('五条', result['strategy'])
        self.assertIn('八条', result['strategy'])
        self.assertNotIn('平和', result['strategy'])
        self.assertNotIn('听牌等', result['direction'])
        self.assertIn('听牌等', result['decision'])
        self.assertIn('五条', result['decision'])
        result = Adviser().recommend(Board(parse('123m456p789s1155z'),
                                          reaction={'tile': '5z', 'seat': 3, 'offered': ['pon']}))
        self.assertIn('跳过', result['strategy'])
        self.assertIn('白', result['strategy'])

    def test_visible_riichi_button_adds_reach_and_selection_only_tenpai(self):
        board = Board(parse('123456m345p67s115z'), riichi_available=True)
        values = candidates(board)
        reach = [v for v in values if v.get('action', {}).get('type') == 'reach']
        self.assertTrue(reach)
        self.assertTrue(all(v['sh'] == 0 for v in reach))
        payload = request(board, values)
        self.assertTrue(any(v['type'] == 'reach' for v in payload['questions']['action']['criteria'].values()))
        board.riichi_selecting = True
        self.assertTrue(all(v['sh'] == 0 and 'action' not in v for v in candidates(board)))
        board.riichi_selecting = False
        board.riichi_available = False
        self.assertFalse(any('action' in v for v in candidates(board)))

    def test_bamboo_names_use_tiao(self):
        self.assertEqual(name('1s'), '一条')
        self.assertEqual(name('0s'), '赤五条')

    def test_screenshot_calls_only_use_offered_actions_and_owned_tiles(self):
        board = Board(parse('123m456p789s1155z'), reaction={'tile': '5z', 'seat': 3, 'offered': ['pon']})
        actions = reaction_candidates(board)
        self.assertEqual({v['action']['type'] for v in actions}, {'pon', 'none'})
        self.assertEqual(next(v for v in actions if v['action']['type'] == 'pon')['action']['consumed'], ['P', 'P'])
        board.reaction = {'tile': '2m', 'seat': 3, 'offered': ['chi']}
        self.assertEqual({v['action']['type'] for v in reaction_candidates(board)}, {'chi', 'none'})
        board.reaction['seat'] = 1
        self.assertEqual([v['action']['type'] for v in reaction_candidates(board)], ['none'])

    def test_split_visible_open_melds(self):
        from desktop.jev_sidecar.screen_state import split_melds
        self.assertEqual(len(split_melds(parse('123m555p'), 2)), 2)
        self.assertIsNone(split_melds(parse('124m'), 1))
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

    def test_immediate_call_discard_respects_kuikae(self):
        board = Board(parse('14m456p789s115z'), melds=[parse_melds('123m'), [], [], []], forbidden_discards=['1m', '4m'])
        self.assertFalse({'1m', '4m'} & {v['tile'] for v in candidates(board)})

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
    def test_five_bamboo_red_center_is_not_a_red_five(self):
        image = Image.new('RGB', (100, 140), '#f0efe6')
        draw = ImageDraw.Draw(image)
        for x, y in [(20, 15), (65, 15), (20, 85), (65, 85)]:
            draw.rectangle((x, y, x + 12, y + 35), fill='#176337')
        draw.rectangle((43, 50, 55, 85), fill='#b82024')
        self.assertEqual(red_five(image, Detection('5s', .99, (0, 0, 100, 140))), '5s')
        for x, y in [(20, 15), (65, 15), (20, 85), (65, 85)]:
            draw.rectangle((x, y, x + 12, y + 35), fill='#b82024')
        self.assertEqual(red_five(image, Detection('5s', .99, (0, 0, 100, 140))), '0s')
    def test_call_transition_tracks_meld_when_sideways_face_is_not_detected(self):
        from desktop.jev_sidecar.screen_state import ScreenStateReader
        from unittest.mock import Mock
        hand = parse('1135779m3667p27z')
        event = {'seat': 2, 'tile': '1m', 'offered': ['pon']}
        image = Image.new('RGB', (400, 200))
        regions = {'hand': {'rect': [0, 0, 1, 1]}}
        detector = Mock()
        detector.detect.return_value = []
        found = lambda tiles: {'hand': [Detection(t, .9, (0, 0, 10, 20)) for t in tiles]}
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / 'screen-state.json'
            reader = ScreenStateReader(state_path)
            with patch.object(reader, 'buttons', return_value=['pon']), patch.object(reader, 'highlighted_reaction', return_value=event):
                reader.read(image, regions, detector, found(hand))
            with patch.object(reader, 'buttons', return_value=[]):
                reader.read(image, regions, detector, found(hand))
            remaining = hand[2:]
            result = reader.read(image, regions, detector, found(remaining))
            self.assertEqual(result['melds'], parse_melds('111m'))
            self.assertEqual(result['forbidden'], ['1m'])
            detector.detect.return_value = [Detection('4m', .9, (i * 20, 0, i * 20 + 18, 20)) for i in range(3)]
            self.assertEqual(reader.read(image, regions, detector, found(remaining))['melds'], parse_melds('111m'))
            restored = ScreenStateReader(state_path)
            self.assertEqual(restored.read(image, regions, detector, found(remaining))['melds'], parse_melds('111m'))

    def test_complete_hand_keeps_raised_and_separated_drawn_tiles(self):
        hand = parse('11379m3667p1s237z5m')
        detected = [Detection(t, .9, (i * 40, 0 if i == 9 else 20, i * 40 + 35, 130 if i == 9 else 150))
                    for i, t in enumerate(hand)]
        detector = TileDetector('unused.onnx')
        with patch.object(detector, 'detect', return_value=detected), patch('desktop.jev_sidecar.vision.hand_boxes') as slots:
            result = detector.regions(Image.new('RGB', (600, 180)), {'hand': {'rect': [0, 0, 1, 1]}})
        self.assertEqual([d.tile for d in result['hand']], hand)
        slots.assert_not_called()

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
    def test_overall_strategy_survives_waiting_and_resets_for_new_hand(self):
        from desktop.jev_sidecar.app import Sidecar
        with tempfile.TemporaryDirectory() as temporary, patch('desktop.jev_sidecar.app.CONFIG', Path(temporary) / 'settings.json'):
            panel = Sidecar(demo=True)
            panel.timer.stop()
            deadline = time.monotonic() + 5
            while panel.jobs and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            first = Adviser().recommend(Board(parse('123456m345p67s115z')))
            panel.show_result(first)
            direction, decision = panel.overall_strategy.text(), panel.explanation.text()
            panel.wait_for_frame('等待摸牌', '当前13张手牌，等待下一次决策')
            self.assertEqual(panel.overall_strategy.text(), direction)
            self.assertEqual(panel.explanation.text(), decision)
            second = Adviser().recommend(Board(parse('11223344556678m')))
            panel.show_result(second)
            self.assertIn('七对子', panel.overall_strategy.text())
            self.assertNotEqual(panel.explanation.text(), decision)
            panel.strategy_hand_id = 0
            panel.recognized(({'hand': [], '_screen': {'hand_id': 1}}, False))
            self.assertIn('新一局', panel.overall_strategy.text())
            self.assertIsNone(panel.current_decision)
            panel.close()

    def test_riichi_second_click_runs_on_partial_selection_without_new_inference(self):
        from desktop.jev_sidecar.app import Sidecar
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temporary, patch('desktop.jev_sidecar.app.CONFIG', Path(temporary) / 'settings.json'):
            panel = Sidecar()
            panel.timer.stop()
            panel.auto_execute.setChecked(True)
            panel.autoplayer.clicker = Mock()
            hand = parse('46m12305667799p3z')
            panel.riichi_flow.begin(hand, '3z')
            panel.frame_started = time.monotonic() + .01
            detections = [Detection(t, 1, (i * 20, 0, i * 20 + 18, 30)) for i, t in enumerate(hand) if t != '0p']
            observation = ({'hand': detections, '_image_size': (300, 100),
                            '_screen': {'riichi_selecting': True, 'melds': [], 'offered': []}}, False)
            panel.recognized(observation)
            panel.recognized(observation)
            self.assertEqual(panel.autoplayer.clicker.call_count, 1)
            self.assertIn('立直第2步', panel.hero.text())
            self.assertNotIn('analysis', panel.busy)
            panel.close()

    def test_main_window_is_compact_and_calibration_is_in_settings(self):
        from desktop.jev_sidecar.app import Sidecar
        with tempfile.TemporaryDirectory() as temporary, patch('desktop.jev_sidecar.app.CONFIG', Path(temporary) / 'settings.json'):
            panel = Sidecar(demo=True)
            panel.show()
            self.app.processEvents()
            self.assertLess(panel.height(), 500)
            self.assertFalse(panel.preview.isVisible())
            self.assertTrue(panel.hero.isVisible())
            panel.open_settings()
            self.app.processEvents()
            self.assertTrue(panel.preview.isVisible())
            panel.settings_dialog.close()
            panel.timer.stop()
            while panel.jobs:
                self.app.processEvents()
                time.sleep(.01)
            panel.close()
    def test_hand_region_includes_drawn_tile_without_repeated_expansion(self):
        from desktop.jev_sidecar.app import include_drawn_tile
        original = {'rect': [.11, .84, .65, .14], 'rotation': 0}
        expanded = include_drawn_tile(original)
        self.assertGreater(expanded['rect'][0] + expanded['rect'][2], .84)
        self.assertLess(expanded['rect'][1], .78)
        self.assertEqual(include_drawn_tile(expanded), expanded)
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

    def test_live_timer_keeps_capturing_but_only_recommends_when_decision_changes(self):
        from desktop.jev_sidecar.app import Sidecar
        with tempfile.TemporaryDirectory() as temporary, patch('desktop.jev_sidecar.app.CONFIG', Path(temporary) / 'settings.json'):
            panel = Sidecar()
            self.assertTrue(panel.live.isChecked())
            self.assertEqual(panel.timer.interval(), 1000)
            panel.capture_rect = [0, 0, 1, 1]
            panel.regions = {'hand': {'rect': [0, 0, 1, 1], 'rotation': 0}}
            hand = parse('12389m278p589s457z')
            detections = [Detection(t, .51, (i * 10, 0, i * 10 + 9, 20)) for i, t in enumerate(hand)]
            recommendation = Adviser().recommend(Board(hand))
            with patch.object(panel, 'screen_image', return_value=Image.new('RGB', (200, 50))) as capture, \
                 patch.object(panel.detector, 'regions', return_value={'hand': detections}), \
                 patch.object(panel.screen_reader, 'read', return_value={'melds': [], 'offered': [], 'reaction': None}), \
                 patch.object(panel.adviser, 'recommend', return_value=recommendation) as recommend:
                deadline = time.monotonic() + 3.5
                while time.monotonic() < deadline and capture.call_count < 3:
                    self.app.processEvents()
                    time.sleep(.01)
                self.assertGreaterEqual(capture.call_count, 3)
                self.assertEqual(recommend.call_count, 1)
                self.assertTrue(all(call.args[1] for call in recommend.call_args_list))
                while panel.jobs:
                    self.app.processEvents()
                    time.sleep(.01)
                self.assertIn('建议打', panel.hero.text())
                self.assertIsNotNone(panel.last_capture)
                self.assertIsNotNone(panel.last_update)
                panel.recognized(({'hand': detections[:-1]}, False))
                self.assertEqual(panel.hero.text(), '等待摸牌')
                self.assertEqual(panel.results.rowCount(), 0)
                panel.recognized(({'hand': detections}, False))
                deadline = time.monotonic() + 2
                while panel.jobs and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(.01)
                self.assertEqual(recommend.call_count, 2, 'A new turn must recalculate even when it draws the same tile')
                panel.recognized(({'hand': []}, False))
                self.assertEqual(panel.hero.text(), '正在寻找手牌')
            panel.timer.stop()
            panel.close()

    def test_selecting_hand_region_starts_automatic_mode(self):
        from desktop.jev_sidecar.app import Sidecar
        with tempfile.TemporaryDirectory() as temporary, patch('desktop.jev_sidecar.app.CONFIG', Path(temporary) / 'settings.json'):
            panel = Sidecar()
            panel.live.setChecked(False)
            panel.lock.setChecked(True)
            panel.capture_rect = [0, 0, 1, 1]
            panel.image = Image.new('RGB', (200, 50))
            with patch('desktop.jev_sidecar.app.RegionDialog') as dialog:
                dialog.return_value.exec.return_value = True
                dialog.return_value.normalized.return_value = [0, 0, 1, 1]
                panel.select_zone()
            self.assertTrue(panel.live.isChecked())
            self.assertFalse(panel.lock.isChecked())
            panel.timer.stop()
            panel.live.setChecked(False)
            panel.close()

    def test_slow_vision_does_not_stop_one_second_screenshots(self):
        from desktop.jev_sidecar.app import Sidecar
        with tempfile.TemporaryDirectory() as temporary, patch('desktop.jev_sidecar.app.CONFIG', Path(temporary) / 'settings.json'):
            panel = Sidecar()
            panel.capture_rect = [0, 0, 1, 1]
            panel.regions = {'hand': {'rect': [0, 0, 1, 1]}}
            hand = parse('12389m278p589s457z')
            def slow_detection(*args):
                time.sleep(1.3)
                return {'hand': [Detection(t, .9, (i * 10, 0, i * 10 + 9, 20)) for i, t in enumerate(hand)]}
            with patch.object(panel, 'screen_image', return_value=Image.new('RGB', (200, 50))) as capture, \
                 patch.object(panel.detector, 'regions', side_effect=slow_detection), \
                 patch.object(panel.screen_reader, 'read', return_value={'melds': [], 'offered': [], 'reaction': None}), \
                 patch.object(panel.adviser, 'recommend', return_value=Adviser().recommend(Board(hand))):
                deadline = time.monotonic() + 3.3
                while time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(.01)
                self.assertGreaterEqual(capture.call_count, 3)
                panel.live.setChecked(False)
                panel.timer.stop()
                while panel.jobs:
                    self.app.processEvents()
                    time.sleep(.01)
            panel.close()


if __name__ == '__main__':
    unittest.main()
