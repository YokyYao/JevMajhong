"""Prepare a prospective within-source call-label balance, not a student fit.

Preserve all thirteen source masses and raw rows. Within the four call/kan
sources, allocate equal loss mass across observed primary action-label classes,
subject to a per-row cap. Missing classes are reported, never fabricated.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import MASS

METHOD = 'prospective_calls13_within_source_observed_action_type_balance_v1'
SOURCES = {f'advanced:{origin}:{phase}' for origin in ('legacy','teacher') for phase in ('call','kan')}


def category(source, operation):
    primary = ('chi','pon') if source.endswith(':call') else ('ankan','daiminkan','kakan')
    return operation if operation in primary else 'other'


def capped_equal_mass(counts, mass, total, cap=16.):
    if (not counts or any(type(n) is not int or n <= 0 for n in counts.values())
            or type(total) is not int or total <= 0 or not 0 < mass <= 1
            or not math.isfinite(cap) or cap <= 0):
        raise ValueError('Positive observed counts and finite recipe required')
    limits = {key:cap*n/total for key,n in counts.items()}
    if sum(limits.values()) < mass-1e-12:
        raise ValueError('Requested source mass is infeasible under the row-weight cap')
    result = {}; remaining = mass; active = set(counts)
    while active:
        equal = remaining/len(active)
        constrained = {key for key in active if limits[key] < equal-1e-15}
        if not constrained:
            result.update({key:equal for key in active}); break
        for key in sorted(constrained):
            result[key] = limits[key]; remaining -= limits[key]
        active -= constrained
    if (abs(sum(result.values())-mass) > 1e-12
            or any(not 0 < value <= limits[key]+1e-12 for key,value in result.items())):
        raise ValueError('The capped class allocation did not preserve source mass')
    return result


def build(records):
    total = len(records)
    if not total: raise ValueError('Original distinct TRAIN rows required')
    ids = [row['_meta']['id'] for row in records]
    if len(set(ids)) != total: raise ValueError('No copied rows or duplicate IDs')
    bins = defaultdict(Counter); labels = []; source_counts = Counter()
    for row in records:
        q = row['questions']['action']; source = q['src'].removeprefix('curriculum2:')
        if source not in MASS or q.get('target') is not None:
            raise ValueError('Exact thirteen-source hard-label calls corpus required')
        operation = q['criteria'][q['label']]['type']
        group = category(source, operation) if source in SOURCES else 'unchanged_source'
        bins[source][group] += 1; source_counts[source] += 1; labels.append((source,group))
    if set(bins) != set(MASS): raise ValueError('All original thirteen sources required')
    by = {}; groups = {}
    for source, counts in sorted(bins.items()):
        masses = capped_equal_mass(counts, MASS[source], total)
        weights = {group:masses[group]*total/n for group,n in counts.items()}
        previous = MASS[source]*total/source_counts[source]
        by[source] = dict(rows=source_counts[source], source_mass=MASS[source],
            within_source_balance_changed=source in SOURCES, class_counts=dict(counts),
            class_mass=masses, class_row_weights=weights, previous_uniform_row_weight=previous,
            missing_primary_classes=sorted(({'chi','pon'} if source.endswith(':call')
                else {'ankan','daiminkan','kakan'})-set(counts)) if source in SOURCES else [])
        groups[source] = weights
    entries = [dict(index=i,id=ids[i],source=source,label_class=group,weight=groups[source][group])
        for i,(source,group) in enumerate(labels)]
    for source in MASS:
        actual = sum(e['weight'] for e in entries if e['source']==source)/total
        if abs(actual-MASS[source]) > 1e-12: raise ValueError('Actual per-row weights changed a source mass')
    return entries,dict(method=METHOD, train_rows=total, sources=by,
        source_mass_unchanged=True, raw_inputs_labels_and_source_partitions_unchanged=True,
        per_original_empirical_objective_intentionally_changed_within_four_sources=True,
        max_row_weight=max(e['weight'] for e in entries), copies_added=0,
        new_contexts_or_expert_labels_added=0, student_updates=0,
        status='prospective_call_label_balance_prepared_not_loss_integration_or_fit_admitted')


def preflight():
    import copy
    import subprocess
    from tools.capture_curriculum_paired_terminal_v2 import resource
    from tools.curriculum_transfer import read, sha
    from tools.curriculum_calls_post_reach_data_v1 import write
    allocation = resource()
    registration = ROOT/'runs/curriculum-calls-classroute-v1/fit-plan/registered.json'
    reg = read(registration)
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    path = Path(args['data']); expected = reg['inputs'].get(str(path.resolve()))
    if expected is None or sha(path) != expected: raise ValueError('Original registered calls TRAIN required')
    records = [json.loads(line) for line in path.open()]
    before = copy.deepcopy(records)
    entries, summary = build(records)
    if len(records) != 49022 or records != before or sha(path) != expected:
        raise ValueError('All49022 original rows must remain unchanged')
    tests = subprocess.run([sys.executable,'-m','unittest','tests.core.test_curriculum_call_type_balance_v1'],
        capture_output=True,text=True,timeout=60)
    if tests.returncode: raise RuntimeError(tests.stdout+tests.stderr)
    out = ROOT/'runs/curriculum-call-type-balance-v1/preflight-v1'
    out.mkdir(parents=True,exist_ok=False)
    write(out/'row-index-and-weights-private.json',entries)
    write(out/'summary-private.json',dict(summary,allocation=allocation,test_output=tests.stdout+tests.stderr,
        validation_and_test_read=False,model_loaded=False,new_games=0,
        prospective_source_index_only=True,live_training_changed=False))
    paths = [path,registration,Path(reg['training_config']['trainer_config']),Path(__file__).resolve(),
        ROOT/'tests/core/test_curriculum_call_type_balance_v1.py',*out.glob('*-private.json')]
    write(out/'completion-binding-private.json',dict(files={str(p):sha(p) for p in paths}))
    print(json.dumps(summary),flush=True)


if __name__ == '__main__': preflight()
