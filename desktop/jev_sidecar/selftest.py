"""Runtime verification executed by the frozen Windows EXE before publication."""
import json
import sys
from pathlib import Path
import time


def model_cases():
    from .engine import Board, parse_melds
    from .tiles import parse
    def board(hand, **kwargs):
        return Board(parse(hand), rivers=[[], [], [], []], metadata_confirmed=True, **kwargs)
    return {
        'closed_discard': board('12389m278p589s457z'),
        'open_discard': board('123m123p123s56z', melds=[parse_melds('777z'), [], [], []]),
        'chi': board('123456m345p67s11z', reaction={'tile': '5s', 'seat': 3, 'offered': ['chi']}),
        'pon': board('123m123p123s1166z', reaction={'tile': '6z', 'seat': 1, 'offered': ['pon']}),
        'ankan': board('0555m123p123s1122z', self_kan_available=True),
        'daiminkan': board('123m123p123s1666z', reaction={'tile': '6z', 'seat': 1, 'offered': ['daiminkan']}),
        'kakan': board('123m123p123s56z', melds=[parse_melds('555z'), [], [], []], self_kan_available=True),
    }


def run(report, model_test=False):
    from PySide6.QtWidgets import QApplication
    from PIL import Image, ImageDraw, ImageFont
    from .app import ROOT, Sidecar, STYLE
    from .engine import Adviser, Board
    from .tiles import parse
    from .vision import TileDetector
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE)
    panel = Sidecar(demo=True)
    panel.show()
    deadline = time.monotonic() + 30
    while panel.jobs and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.02)
    app.processEvents()
    assert '白' in panel.hero.text(), panel.hero.text()
    assert panel.tabs.currentWidget().horizontalScrollBar().maximum() == 0
    detector = TileDetector(ROOT / 'assets/tiles.onnx')
    assert detector.detect(Image.new('RGB', (320, 200), '#174a46')) == []
    from .screen_state import ScreenStateReader
    from .engine import candidates, request
    reader = ScreenStateReader()
    sample = Image.new('RGB', (400, 80), '#18364b')
    draw = ImageDraw.Draw(sample)
    font = ImageFont.truetype(str(ROOT / 'assets/NotoSansSC.ttf'), 44)
    draw.text((45, 2), '×2', font=font, fill='white')
    draw.text((245, 2), '×3', font=font, fill='white')
    counts = reader.counter_reader.read(sample, {'rect': [0, 0, 1, 1]}, reader.ocr_engine)
    assert counts == {'honba': 3, 'sticks': 2}, counts
    panel.update_counters(counts)
    board = panel.board()
    payload = request(board, candidates(board))
    assert (payload['state']['honba'], payload['state']['sticks']) == (3, 2)
    result = {'status': 'passed', 'frozen_exe': bool(getattr(sys, 'frozen', False)), 'gui': True,
              'onnx': True, 'counters': True, 'model': False}
    panel.timer.stop()
    panel.close()
    if model_test:
        import math
        from .unified_policy import UnifiedPolicyClient, default_checkpoint, identity
        adviser = Adviser()
        adviser.load(default_checkpoint(ROOT), device='cpu')
        assert isinstance(adviser.model, UnifiedPolicyClient), 'Jev Play v2 single policy missing from release'
        decisions = {}
        for case, board in model_cases().items():
            prediction = adviser.recommend(board, use_model=True)
            options = prediction['all_candidates']
            probabilities = [v['preference'] for v in options]
            assert all(math.isfinite(p) and p >= 0 for p in probabilities)
            assert abs(sum(probabilities) - 1) < 1e-5
            assert prediction['model_input']['route'] == 'unified'
            assert prediction['engine'] == 'Jev Play v2'
            if case in ('chi', 'pon', 'ankan', 'daiminkan', 'kakan'):
                assert any(v.get('action', {}).get('type') == case for v in options), case
            decisions[case] = prediction['top3'][0].get('action', {'type': 'dahai', 'tile': prediction['top3'][0]['tile']})
        result.update(model=True, model_id='jev-play-v2', device='cpu',
            source_model=identity()['source_model_id'], source_step=identity()['source_step'],
            model_identity_sha256=identity()['identity_sha256'], decisions=decisions)
    Path(report).write_text(json.dumps(result, indent=2), encoding='utf-8')
