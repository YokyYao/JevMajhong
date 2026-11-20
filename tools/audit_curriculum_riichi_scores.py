"""Read-only diagnostics of riichi discrimination and action-type aggregation."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def confusion(labels, decisions):
    c = Counter()
    for truth, actual in zip(labels, decisions):
        c['tp' if truth and actual else 'fn' if truth else 'fp' if actual else 'tn'] += 1
    tpr = c['tp']/(c['tp']+c['fn'])
    tnr = c['tn']/(c['tn']+c['fp'])
    return dict(tp=c['tp'], fn=c['fn'], tn=c['tn'], fp=c['fp'],
                acceptance_recall=tpr, decline_recall=tnr, balanced_accuracy=(tpr+tnr)/2)


def ranking_summary(labels, scores):
    positive = [p for p, y in zip(scores, labels) if y]
    negative = [p for p, y in zip(scores, labels) if not y]
    assert positive and negative
    auc = sum(float(a > b)+.5*float(a == b) for a in positive for b in negative)/(len(positive)*len(negative))
    unique = sorted(set(scores))
    thresholds = [unique[0]-1.]+[(a+b)/2 for a, b in zip(unique, unique[1:])]+[unique[-1]+1.]
    alternatives = [confusion(labels, [p >= t for p in scores]) for t in thresholds]
    return dict(auc=auc, oracle_best_balanced=max(v['balanced_accuracy'] for v in alternatives),
        oracle_both60=any(v['acceptance_recall'] >= .6 and v['decline_recall'] >= .6 for v in alternatives))


def summarize(data, predictions):
    byid = {r['_meta']['id']: r for r in data}
    labels, scores, flat, margins = [], [], [], []
    quality = Counter()
    reach_keys = Counter()
    for p in predictions:
        row = byid[p['id']]
        q = row['questions']['action']
        reach = {k for k, c in q['criteria'].items() if c['type'] == 'reach'}
        if not reach:
            continue
        probabilities = dict(zip(p['keys'], p['p']))
        assert set(probabilities) == set(q['criteria'])
        truth = q['label'] in reach
        score = sum(probabilities[k] for k in reach)
        assert math.isfinite(score) and 0 <= score <= 1.000001
        decision = max(probabilities, key=probabilities.__getitem__) in reach
        labels.append(truth)
        scores.append(min(1., score))
        flat.append(decision)
        margins.append(math.log(max(max(probabilities[k] for k in reach), 1e-30))
            -math.log(max(max(probabilities[k] for k in probabilities if k not in reach), 1e-30)))
        quality['self_already_declared_riichi'] += row['state']['seat'] in row['state']['riichi']
        quality['flat_reach_while_discard_group_mass_larger'] += decision and score < .5
        quality['flat_false_positive_while_discard_group_mass_larger'] += decision and not truth and score < .5
        if not truth:
            sh = q['criteria'][q['label']].get('sh')
            quality['teacher_decline_missing_discard_shanten'] += sh is None
            quality['teacher_decline_keeps_tenpai'] += sh == 0
            quality['teacher_decline_breaks_tenpai'] += isinstance(sh, (int, float)) and sh > 0
        for k in reach:
            reach_keys.update(q['criteria'][k].keys())
    positive = [p for p, y in zip(scores, labels) if y]
    negative = [p for p, y in zip(scores, labels) if not y]
    assert positive and negative
    # This is an oracle diagnostic on consumed development labels, not a threshold to deploy.
    probability_ranking = ranking_summary(labels, scores)
    margin_ranking = ranking_summary(labels, margins)
    return dict(windows=len(labels), teacher_accepts=len(positive), teacher_declines=len(negative),
        flat_argmax=confusion(labels, flat), grouped_type_at_half=confusion(labels, [p >= .5 for p in scores]),
        reach_mass_auc=probability_ranking['auc'], mean_reach_mass_positive=statistics.mean(positive),
        mean_reach_mass_negative=statistics.mean(negative),
        binary_brier=statistics.mean((p-float(y))**2 for p, y in zip(scores, labels)),
        oracle_best_balanced_accuracy_on_consumed_validation=probability_ranking['oracle_best_balanced'],
        oracle_any_threshold_with_both_recalls_at_least_06=probability_ranking['oracle_both60'],
        flat_logit_margin_auc=margin_ranking['auc'],
        oracle_flat_margin_best_balanced_accuracy=margin_ranking['oracle_best_balanced'],
        oracle_flat_margin_both_recalls_at_least_06=margin_ranking['oracle_both60'],
        no_threshold_chosen_or_deployed=True, data_quality=dict(quality), reach_branch_field_counts=dict(reach_keys))


def audit(registered, comparison, out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    runs = []
    expected = None
    for file in ([registered, comparison] if comparison else [registered]):
        reg = read(file)
        ready = read(ROOT/reg['curriculum']['data_root']/'data-ready.json')
        spec = ready['stages']['stage1']['validation']
        path = ROOT/spec['path']
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        assert actual == spec['sha256']
        if expected is None:
            expected = actual
        assert expected == actual, 'The two runs must use the exact same consumed validation set'
        data = records(path)
        fit = ROOT/reg['curriculum']['out']
        points = []
        for point in read(fit/'validation-curve.json'):
            predictions = read(fit/'training-run/training_eval'/f"step-{point['step']:06d}"/'development/rows.json')
            points.append(dict(step=point['step'], **summarize(data, predictions)))
        runs.append(dict(run_id=reg['curriculum']['run_id'], points=points))
    result = dict(status='riichi_saved_score_discrimination_audit_complete', node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'], validation_sha256=expected, runs=runs,
        no_new_inference_or_games=True, no_optimizer_updates=True, no_policy_changed=True,
        no_threshold_selected=True,
        caveat='Consumed development labels only. AUC/oracle thresholds diagnose discrimination and arity effects, not independent strength or expert optimality. Do not deploy a validation-selected threshold or claim a causal explanation from these summaries.')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--comparison', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit(a.registered, a.comparison, a.out)
