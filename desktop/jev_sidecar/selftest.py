"""Runtime verification executed by the frozen Windows EXE before publication."""
import json
import sys
from pathlib import Path
import time


def run(report, model_test=False):
    from PySide6.QtWidgets import QApplication
    from PIL import Image
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
    result = {'status': 'passed', 'frozen_exe': bool(getattr(sys, 'frozen', False)), 'gui': True, 'onnx': True, 'model': False}
    panel.timer.stop()
    panel.close()
    if model_test:
        adviser = Adviser()
        adviser.load(ROOT / 'models/calibration-02', device='cpu')
        board = Board(parse('12389m278p589s457z'), rivers=[[], [], [], []], metadata_confirmed=True)
        prediction = adviser.recommend(board, use_model=True)
        assert sum(v['eligible'] for v in prediction['all_candidates']) >= 2
        assert prediction['top3'][0]['tile'] in board.hand
        result.update(model=True, chosen=prediction['top3'][0]['tile'], device='cpu')
    Path(report).write_text(json.dumps(result, indent=2), encoding='utf-8')
