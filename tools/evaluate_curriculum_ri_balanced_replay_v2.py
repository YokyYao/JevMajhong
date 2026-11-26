"""V2 run validation with the unchanged V1 complete-East evaluator."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_ri_balanced_replay_v2 import evaluate

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--step', type=int, required=True)
    a = p.parse_args()
    evaluate(a.registered.resolve(), a.checkpoint.resolve(), a.step)
