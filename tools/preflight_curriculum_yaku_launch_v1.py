"""One bounded integration check of the real 3B corpus with synthetic outputs."""
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_yaku_objective_v1 import MASS, METHOD
from tools.curriculum_yaku_fit_v1 import profile
from tools.register_curriculum_yaku_course_v1 import PUBLIC_FILES


def preflight():
    gpu_only()
    out = ROOT/'runs/curriculum-yaku-course-v1/launch-preflight-v1'; out.mkdir(parents=True, exist_ok=False)
    config = ROOT/'configs/training/curriculum-yaku-fit-v1.json'; c = read(config)
    parent = read(ROOT/c['parent_registration'])
    sys.path.insert(0, parent['source']); sys.path.insert(0, parent['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevmahjong.training.base_lora import target_cross_entropy
    data_path = ROOT/c['data_ready']; data = read(data_path)
    binding = read(ROOT/'runs/curriculum-yaku-course-v1/prospective-recipe-v1/completion-binding-private.json')
    files = dict(binding['files'])
    for name in PUBLIC_FILES:
        path = ROOT/name
        if name.endswith('.py'): compile(path.read_text(), str(path), 'exec')
        files[str(path.resolve())] = sha(path)
    path = ROOT/data['files']['validation']['path']
    if sha(path) != data['files']['validation']['sha256']: raise ValueError('Actual validation changed')
    files[str(path.resolve())] = sha(path); files[str(data_path)] = sha(data_path)
    validation = [json.loads(line) for line in path.open()]
    predictions = []; objectives = defaultdict(list); ces = defaultdict(list)
    for row in validation:
        q = row['questions']['action']; keys = list(q['criteria']); n = len(keys)
        predictions.append(dict(id=row['_meta']['id'], keys=keys, p=[1/n]*n))
        kinds = {v['type'] for v in q['criteria'].values()}
        if q.get('target') is not None:
            objective = ce = math.log(sum(v > 0 for v in q['target'].values()))
        elif kinds == {'reach', 'dahai'}:
            yes = q['criteria'][q['label']]['type'] == 'reach'
            objective = 2*math.log(n) if yes else 2*math.log(n/(n-1))+math.log(n-1)
            ce = math.log(n)
        else:
            objective = ce = math.log(n)
        objectives[q['src']].append(objective); ces[q['src']].append(ce)
    actual = profile(validation, predictions, {}, target_cross_entropy)
    expected = sum(MASS[s]*sum(v)/len(v) for s, v in objectives.items())
    expected_ce = sum(MASS[s]*sum(v)/len(v) for s, v in ces.items())
    error = abs(actual['weighted_class_objective']-expected)
    ce_error = abs(actual['weighted_target_ce']-expected_ce)
    expected_replay = {s: n for s, n in data['coverage']['validation']['sources'].items()
        if not s.startswith('curriculum3B:')}
    if (error > 1e-10 or ce_error > 1e-10 or sum(actual['stratum_counts'].values()) != 24516
            or len(actual['new_source_teacher_agreement']) != 6
            or actual['replay_profile']['stratum_counts'] != expected_replay):
        raise ValueError('Complete25-source synthetic analytic profile integration failed')
    # This result is deliberately excluded from all student loss/game curves.
    write(out/'synthetic-profile-private.json', actual)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('An immutable preparation input changed')
    result = dict(status='yaku_launch_full_synthetic_validation_integration_passed_not_parent_or_fit_admitted',
        method=METHOD, actual_validation_rows=24516, original_replay_rows=23688, new_route_rows=828,
        analytic_objective_error=error, analytic_CE_error=ce_error,
        actual_optimizer_steps=0, model_loaded=False, synthetic_profile_forbidden_for_student_curves=True,
        original_3A_launch_unchanged=True, job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
    write(out/'completion-binding-private.json', dict(files=files))
    write(out/'summary-private.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': preflight()
