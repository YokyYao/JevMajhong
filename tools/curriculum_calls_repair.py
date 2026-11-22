"""Twelve-source calls metrics and a source-qualified basic warm-start bridge.

This module admits no unfinished parent and changes no model input or action
selection. Its basic-only projection keeps the original six-source controls
visible when the later course adds many more call/PASS examples.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path

from tools.build_curriculum_calls_repair import MASS
from tools.curriculum_basic_repair import MASS as BASIC_MASS, grade, profile as basic_profile
from tools.curriculum_parent import ParentNotReady, require_replay
from tools.curriculum_transfer import METHOD, ROOT, read, sha, verify_curriculum_parent
from tools.seal_curriculum_basic_repair import require_source_quality

OPERATIONS = {'chi', 'pon', 'ankan', 'daiminkan', 'kakan', 'reach'}


def key(row):
    source = row['questions']['action']['src']
    if not source.startswith('curriculum2:') or source[len('curriculum2:'):] not in MASS:
        raise ValueError('An explicit registered twelve-source category is required')
    return source[len('curriculum2:'):]


def prediction_index(records, predictions):
    byid = {r['_meta']['id']: r for r in records}
    predicted = {p['id']: p for p in predictions}
    if len(byid) != len(records) or len(predicted) != len(predictions) or set(byid) != set(predicted):
        raise ValueError('Prediction IDs must cover every distinct record exactly once')
    for identifier, row in byid.items():
        q = row['questions']['action']; p = predicted[identifier]
        if (len(p['keys']) != len(p['p']) or len(set(p['keys'])) != len(p['keys'])
            or set(p['keys']) != set(q['criteria']) or q['label'] not in q['criteria']
            or q.get('target') is not None):
            raise ValueError('Exact hard-target action menus are required')
        if (any(not math.isfinite(v) or v < 0 for v in p['p'])
            or abs(sum(p['p']) - 1) > 1e-5):
            raise ValueError('Invalid saved action probability distribution')
    return predicted


def basic_projection(records):
    """Change only the private loss tag, retaining original public inputs/labels."""
    projected = []
    for row in records:
        category = key(row)
        if category.startswith('basic:'):
            item = copy.deepcopy(row)
            item['questions']['action']['src'] = 'curriculum1:' + category[len('basic:'):]
            projected.append(item)
    if {r['questions']['action']['src'].removeprefix('curriculum1:') for r in projected} != set(BASIC_MASS):
        raise ValueError('All six basic replay strata must remain available')
    return projected


def profile(records, predictions, mass, base, ce):
    if mass != MASS:
        raise ValueError('Prospective twelve-source loss masses changed')
    saved = prediction_index(records, predictions)
    ces = defaultdict(list); hits = defaultdict(list)
    ops = defaultdict(Counter); source_ops = defaultdict(lambda: defaultdict(Counter))
    for row in records:
        category = key(row); q = row['questions']['action']; p = saved[row['_meta']['id']]
        chosen = p['keys'][max(range(len(p['p'])), key=p['p'].__getitem__)]
        ces[category].append(ce(q, p['keys'], p['p'])); hits[category].append(chosen == q['label'])
        truth = q['criteria'][q['label']]['type']; actual = q['criteria'][chosen]['type']
        source = ':'.join(category.split(':')[:2])
        for operation in {v['type'] for v in q['criteria'].values()} & OPERATIONS:
            yes, selected = truth == operation, actual == operation
            outcome = 'tp' if yes and selected else 'fn' if yes else 'fp' if selected else 'tn'
            ops[operation][outcome] += 1; source_ops[source][operation][outcome] += 1
    if set(ces) != set(MASS):
        raise ValueError('Incomplete twelve-source validation')
    basic = basic_projection(records)
    bp = [saved[r['_meta']['id']] for r in basic]
    natural = sum(ce(r['questions']['action'], p['keys'], p['p']) for r, p in zip(basic, bp)) / len(basic)
    replay = basic_profile(basic, bp, BASIC_MASS, {'weighted_target_ce': natural}, ce)
    means = {k: sum(v)/len(v) for k, v in ces.items()}
    convert = lambda values: {k: {c: v[c] for c in ('tp', 'fn', 'tn', 'fp')} for k, v in values.items()}
    return dict(base, phase_weighted_ce=base['weighted_target_ce'],
        weighted_target_ce=sum(mass[k]*means[k] for k in mass),
        natural_target_ce=sum(sum(v) for v in ces.values())/len(records),
        stratum_ce=means, stratum_accuracy={k: sum(v)/len(v) for k, v in hits.items()},
        stratum_counts={k: len(v) for k, v in ces.items()},
        basic_replay=replay, basic_replay_source_quality=grade(replay),
        riichi_type_counts=replay['riichi_type_counts'], riichi_by_source=replay['riichi_by_source'],
        operation_type_counts=convert(ops),
        operation_type_counts_by_source={s: convert(v) for s, v in source_ops.items()},
        action_agreement_is_not_policy_strength=True)


def require_parent_prediction_parity(parent_rows, parent_predictions, child_rows, child_predictions, tolerance=5e-5):
    """Check every actual basic validation row at the child's saved step zero."""
    if not math.isfinite(tolerance) or tolerance < 0 or tolerance > 5e-5:
        raise ValueError('Warm-start probability tolerance cannot be relaxed')
    old = prediction_index(parent_rows, parent_predictions)
    current = prediction_index(child_rows, child_predictions)
    basic = basic_projection(child_rows)
    if {r['_meta']['id'] for r in basic} != set(old):
        raise ValueError('Basic validation subset differs from the completed parent')
    require_replay(parent_rows, basic)
    error = 0.
    for identifier, previous in old.items():
        actual = current[identifier]
        if actual['keys'] != previous['keys']:
            raise ValueError('Warm-start validation changed candidate order')
        error = max(error, max(abs(a-b) for a, b in zip(actual['p'], previous['p'])))
    if error > tolerance:
        raise ValueError('The child step-zero basic predictions differ from its parent')
    return dict(checked_basic_menus=len(old), max_probability_error=error, tolerance=tolerance,
        public_inputs_and_labels_preserved=True, not_a_tensor_load_or_parent_admission_proof=True)


def verify_source_qualified_parent(cfg):
    """Wrap the full existing model/lineage verifier with the stronger source gate."""
    spec = cfg['parent_initialization']; seal_path = Path(spec['manifest']).resolve()
    if sha(seal_path) != spec['sha256']:
        raise ParentNotReady('Source-qualified parent seal changed')
    seal = read(seal_path)
    if (spec['method'] != METHOD or seal.get('method') != METHOD
        or seal.get('source_specific_parent_quality_checked') is not True
        or seal.get('source_specific_sealer') != 'tools/seal_curriculum_basic_repair.py'):
        raise ParentNotReady('A completed source-qualified basic parent is mandatory; no research override')
    # The original verifier checks bound model, protocol, exposure/reload evidence,
    # source package, serving schema, exact child data and new optimizer setup.
    verified = verify_curriculum_parent(cfg)
    reg = read(seal['parent_registration']); evidence = read(seal['parent_evidence'])
    fit = ROOT/reg['curriculum']['out']
    validation = [json.loads(s) for s in (fit/'metrics/validation_metrics.jsonl').open()][-1]
    quality = require_source_quality(reg, evidence, validation)
    if quality != seal.get('basic_repair_source_quality'):
        raise ParentNotReady('Parent source-quality seal differs from audited final metrics')
    for name in ('tools/seal_curriculum_basic_repair.py', 'tools/curriculum_basic_repair.py'):
        path = (ROOT/name).resolve()
        if seal['bound_files'].get(str(path)) != sha(path):
            raise ParentNotReady('The actual source-quality implementation must remain sealed')
    return dict(verified, source_specific_basic_quality_verified=True)


@contextmanager
def source_qualified_parent_loader(initialization_module):
    original = initialization_module.verified_parent
    initialization_module.verified_parent = verify_source_qualified_parent
    try:
        yield
    finally:
        initialization_module.verified_parent = original
