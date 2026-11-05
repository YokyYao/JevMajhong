"""Exercise the actual shipped checkpoint through the desktop backend on a GPU."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from desktop.jev_sidecar.engine import Adviser, Board, CHECKPOINT_SHA
from desktop.jev_sidecar.tiles import parse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'runs/head-rl-diagonal-value-20261030/checkpoints/calibration-02')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Preserve existing model smoke result')
    adviser = Adviser()
    start = time.monotonic()
    loaded = adviser.load(args.checkpoint)
    load_seconds = time.monotonic() - start
    board = Board(parse('123456m345p67s115z'), rivers=[[], [], [], []], indicators=parse('4m'), metadata_confirmed=True)
    cases = []
    for hand in ['123456m345p67s115z', '12389m278p589s457z', '12389m278p589s457z']:
        board.hand = parse(hand)
        start = time.monotonic()
        result = adviser.recommend(board, use_model=True)
        seconds = time.monotonic() - start
        assert result['engine'] == 'Jev 校正版2'
        assert all(r['tile'] in board.hand and r['eligible'] and 0 <= r['preference'] <= 1 for r in result['top3'])
        assert abs(sum(v['preference'] for v in result['all_candidates']) - 1) < 1e-5
        cases.append({'inference_seconds': seconds, 'eligible_candidates': sum(v['eligible'] for v in result['all_candidates']),
                      'recommended_tile': result['top3'][0]['tile']})
    assert cases[1]['eligible_candidates'] >= 2, 'Exercise a real model choice among multiple guarded discards'
    import torch
    record = {'status': 'passed', 'head_sha256': CHECKPOINT_SHA, 'loader': loaded,
              'load_seconds': load_seconds, 'cases': cases,
              'peak_allocated_vram_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
              'recommended_tile': result['top3'][0]['tile'], 'guarded_legal_hand_tile': True,
              'platform_tested': sys.platform, 'windows_runtime_verified': False,
              'screenshot_domain_playing_strength_verified': False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False))
