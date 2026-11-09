"""Real local GPU checks for call scoring and post-call discard scoring."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from desktop.jev_sidecar.engine import Adviser, Board, parse_melds
from desktop.jev_sidecar.tiles import parse


if __name__ == '__main__':
    adviser = Adviser()
    loaded = adviser.load(ROOT / 'models/calibration-02')
    cases = [
        Board(parse('123m456p789s1155z'), reaction={'tile': '5z', 'seat': 1, 'offered': ['pon']}),
        Board(parse('123m456p789s1155z'), reaction={'tile': '2m', 'seat': 3, 'offered': ['chi']}),
        Board(parse('123m456p789s11z'), melds=[parse_melds('555z'), [], [], []]),
    ]
    records = []
    for board in cases:
        start = time.perf_counter()
        result = adviser.recommend(board, use_model=True)
        elapsed = time.perf_counter() - start
        assert abs(sum(v['preference'] for v in result['all_candidates']) - 1) < 1e-5
        if board.reaction:
            assert {v['action']['type'] for v in result['all_candidates']} == {board.reaction['offered'][0], 'none'}
        else:
            assert result['top3'][0]['tile'] in board.hand
        records.append({'seconds': elapsed, 'reaction': board.reaction, 'top3': result['top3']})
    report = {'status': 'passed', 'loader': loaded, 'cases': records,
              'screenshot_call_detection_verified': False}
    out = ROOT / 'runs/live-debug/gpu-calls.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))
