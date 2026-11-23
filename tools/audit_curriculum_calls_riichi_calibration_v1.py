"""Cached-only RI ranking/calibration at registered calls0/194 points."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import socket
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write


def diagnostic(values):
    positive = [p for y, p in values if y]; negative = [p for y, p in values if not y]
    bins = []
    for index in range(10):
        items = [(y, p) for y, p in values if min(int(p*10), 9) == index]
        if items:
            bins.append(dict(lower=index/10, upper=(index+1)/10, count=len(items),
                predicted_mean=sum(p for _, p in items)/len(items), observed_accept_fraction=sum(y for y, _ in items)/len(items)))
    n = len(values)
    if not n: raise ValueError('Real observed RI values required')
    auc = (sum(float(a > b)+.5*float(a == b) for a in positive for b in negative)/(len(positive)*len(negative))
           if positive and negative else None)
    tp = sum(y and p >= .5 for y, p in values); tn = sum(not y and p < .5 for y, p in values)
    return dict(rows=n, accept=len(positive), decline=len(negative),
        roc_auc=auc, brier=sum((p-y)**2 for y, p in values)/n,
        mean_predicted_accept=sum(p for _, p in values)/n, observed_accept_fraction=len(positive)/n,
        fixed_0_5_tp=tp, fixed_0_5_fn=len(positive)-tp, fixed_0_5_tn=tn, fixed_0_5_fp=len(negative)-tn,
        fixed_0_5_balanced_accuracy=(tp/len(positive)+tn/len(negative))/2 if positive and negative else None,
        equal_width_10bin_ece=sum(b['count']*abs(b['predicted_mean']-b['observed_accept_fraction']) for b in bins)/n,
        bins=bins)


def audit(out):
    gpu_only()
    registered = ROOT/'runs/curriculum-calls-classroute-v1/fit-plan/registered.json'; reg = read(registered)
    c = reg['curriculum']; ready = read(c['data_ready']); spec = ready['files']['validation']; data_path = ROOT/spec['path']
    if sha(data_path) != spec['sha256'] or spec['records'] != 12479: raise ValueError('Frozen calls validation changed')
    from tools.curriculum_calls_repair import prediction_index
    records = [json.loads(line) for line in data_path.open()]
    basic = [r for r in records if r['questions']['action']['src'] in {
        'curriculum2:basic:legacy:riichi_yes', 'curriculum2:basic:legacy:riichi_no',
        'curriculum2:basic:teacher:riichi_yes', 'curriculum2:basic:teacher:riichi_no'}]
    if len(basic) != 632: raise ValueError('Unchanged632 basic RI windows required')
    paths = [registered, data_path, Path(__file__), Path(c['data_ready']),
        ROOT/'tools/curriculum_calls_repair.py', ROOT/'cluster/curriculum-calls-riichi-calibration-v1.slurm',
        ROOT/'tests/integration/test_curriculum_calls_riichi_calibration_v1.py',
        ROOT/'docs/experiments/plans/curriculum-calls-riichi-calibration-20261123.md']
    fit = ROOT/c['out']; snapshots = ROOT/'runs/curriculum-calls-curve-step194-v1/snapshot-v1/snapshots-private'
    validation_path = snapshots/'validation_metrics.jsonl'; paths.append(validation_path)
    val = {r['local_step']: r for r in (json.loads(line) for line in validation_path.open()) if r['local_step'] <= 194}
    if set(val) != {0, 194}: raise ValueError('Actual complete cached0/194 source validation required')
    for step in (0, 194): paths.append(fit/'training-run/training_eval'/f'step-{step:06d}'/'development/rows.json')
    files = {str(p.resolve()): sha(p) for p in paths}
    active = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files']
    out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(files=files, steps=[0, 194], actual_basic_RI_rows=632,
        threshold_fixed=.5, probability='sum of full legal reach candidate mass',
        no_threshold_sweep_model_calls_games_updates=True, grouping='source and zero/nonzero opponent RI'))
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests/integration'), pattern='test_curriculum_calls_riichi_calibration_v1.py')
    tested = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tested.wasSuccessful() or tested.testsRun != 2: raise ValueError('Independent numerical diagnostic checks failed')
    results = {}
    for step in (0, 194):
        predictions = prediction_index(records, read(fit/'training-run/training_eval'/f'step-{step:06d}'/'development/rows.json'))
        groups = defaultdict(list); families = defaultdict(set)
        for row in basic:
            q = row['questions']['action']; source = q['src'].split(':')[2]; label = q['criteria'][q['label']]['type'] == 'reach'
            p = predictions[row['_meta']['id']]; types = {key: q['criteria'][key]['type'] for key in p['keys']}
            probability = sum(v for key, v in zip(p['keys'], p['p']) if types[key] == 'reach')
            if not math.isfinite(probability) or not 0 <= probability <= 1: raise ValueError('Finite full-class RI mass required')
            threats = sum(seat != row['state']['seat'] for seat in row['state']['riichi'])
            for name in ('pooled', source, source+(':opponent_RI' if threats else ':no_opponent_RI')):
                groups[name].append((int(label), probability)); families[name].add(row['_meta']['group_id'])
        result = {name: dict(diagnostic(values), source_groups=len(families[name])) for name, values in groups.items()}
        for source in ('legacy', 'teacher'):
            expected = val[step]['basic_replay']['riichi_by_source'][source]; actual = result[source]
            if any(actual['fixed_0_5_'+key] != expected.get(key, 0) for key in ('tp', 'fn', 'tn', 'fp')):
                raise ValueError('Cached class metrics differ from actual frozen-route validation')
        results[str(step)] = result
    if not all(sha(p) == h for p, h in files.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Bound cached or active training artifact changed')
    write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active))
    summary = dict(status='cached_calls_riichi_ranking_and_calibration_diagnosed', results=results,
        numerical_tests=2,
        all_cached_class_counts_match_actual_validation=True,
        cached_values_not_native_guard_replays=True, no_threshold_fitted_or_selected=True,
        bins_and_auc_are_descriptive_not_strategy_or_population_proof=True,
        new_model_calls=0, new_games=0, student_updates=0, active_fit_dependencies_unchanged=len(active),
        raw_record_probabilities_and_labels_private=True, job_id=os.environ['SLURM_JOB_ID'],
        node=socket.gethostname(), goal_achieved=False)
    write(out/'summary-private.json', summary)
    print(json.dumps(dict(status=summary['status'], results={step: {name: {k: v for k, v in d.items() if k != 'bins'}
        for name, d in values.items()} for step, values in results.items()})), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); audit(a.out.resolve())
