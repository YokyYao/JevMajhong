"""Runtime verification executed by the frozen Windows EXE before publication."""
import json
import sys
from pathlib import Path
import time


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
        adviser = Adviser()
        adviser.load(ROOT / 'models/calibration-02', device='cpu')
        board = Board(parse('12389m278p589s457z'), rivers=[[], [], [], []], metadata_confirmed=True)
        prediction = adviser.recommend(board, use_model=True)
        assert sum(v['eligible'] for v in prediction['all_candidates']) >= 2
        assert prediction['top3'][0]['tile'] in board.hand
        from .stage2_runtime import Stage2Client
        assert isinstance(adviser.model, Stage2Client), 'Stage2 call adapter missing from release'
        call_board = Board(parse('123456m345p67s11z'), rivers=[[], [], [], []], metadata_confirmed=True,
            reaction={'tile': '5s', 'seat': 3, 'offered': ['chi']})
        call_prediction = adviser.recommend(call_board, use_model=True)
        assert call_prediction['reaction'] and len(call_prediction['all_candidates']) >= 2
        result.update(model=True, chosen=prediction['top3'][0]['tile'], device='cpu',
            call_model='feature-stage2-step200', ordinary_model='calibration-02',
            call_choice=call_prediction['top3'][0]['action'])
    Path(report).write_text(json.dumps(result, indent=2), encoding='utf-8')
