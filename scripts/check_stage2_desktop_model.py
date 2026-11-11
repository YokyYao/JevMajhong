"""Allocated-GPU parity of the restored desktop mixture, without GUI or screenshots."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from head_rl import write


def main():
    import torch
    from desktop.jev_sidecar.engine import Adviser, Board
    from desktop.jev_sidecar.tiles import parse
    from desktop.jev_sidecar.stage2_runtime import Stage2Client
    from jevany.data import api_request
    out = ROOT / 'runs/stage2-desktop-model-20261111'
    out.mkdir(parents=True, exist_ok=False)
    adviser = Adviser(); adviser.load(ROOT / 'models/calibration-02')
    if not isinstance(adviser.model, Stage2Client): raise ValueError('Desktop call model not loaded')
    source = ROOT / 'runs/feature-unseen-continuation-20261110'
    rows = json.loads((source / 'training-run/training_eval/step-000200/development/rows.json').read_text())
    index = {r['id']: r for r in rows}
    data = ROOT / 'data/feature-unseen-continuation-20261110'
    count = ordinary = 0; maximum = 0.
    for line in (data / 'validation.jsonl').open():
        record = json.loads(line)
        if record['_meta']['phase'] != 'reaction' or record['_meta']['view'] != 'native': continue
        request = api_request(record); expected = index[record['_meta']['id']]
        keys = list(request['questions']['action']['criteria'])
        p = adviser.model.system_one(**request)['answers']['action']['probabilities']
        if keys != list(p) or keys != expected['keys']: raise ValueError('Desktop native option order changed')
        actual, target = torch.tensor([p[k] for k in keys]), torch.tensor(expected['p'])
        error = float((actual - target).abs().max()); maximum = max(maximum, error)
        if error > 2e-5 or actual.argmax() != target.argmax(): raise ValueError('Desktop stage2 probability mismatch')
        count += 1
        if count == 12: break
    for line in (data / 'train.jsonl').open():
        record = json.loads(line)
        if record['_meta']['phase'] == 'reaction': continue
        request = api_request(record)
        before = adviser.model.ordinary.system_one(**request)['answers']['action']['probabilities']
        after = adviser.model.system_one(**request)['answers']['action']['probabilities']
        if before != after: raise ValueError('Desktop oldB ordinary route changed')
        ordinary += 1
        if ordinary == 6: break
    if count != 12 or ordinary != 6: raise ValueError('Insufficient parity coverage')
    board = Board(parse('123456m345p67s11z'), rivers=[[], [], [], []], metadata_confirmed=True,
        reaction={'tile': '5s', 'seat': 3, 'offered': ['chi']})
    prediction = adviser.recommend(board, use_model=True)
    if not prediction['reaction'] or len(prediction['all_candidates']) < 2:
        raise ValueError('Desktop call recommendation not exercised')
    result = {'status': 'passed_native_model_parity', 'native_reaction_requests': count,
        'ordinary_requests_unchanged': ordinary, 'maximum_probability_error': maximum,
        'decisions_equal': True, 'restored_inference_files_validated': True,
        'desktop_call_request_inference_passed': True, 'screenshot_or_windows_validation': False,
        'human_test_opened': False, 'job_id': os.environ.get('SLURM_JOB_ID')}
    write(out / 'result.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
