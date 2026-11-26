"""Thirteen-source calls loss and explicitly RI-class-selected validation."""
from collections import Counter, defaultdict
import copy
import math

from tools.curriculum_calls_post_reach_data_v1 import MASS, POST, READY, layout
from tools.curriculum_calls_riichi_retention_v1 import annotate as old_annotate, parts as old_parts
from tools.curriculum_riichi_class_loss import FIELD
from tools.curriculum_riichi_control import cached_parts, COMPONENTS as BASIC_COMPONENTS
from tools.curriculum_riichi_class_policy_v1 import METHOD as ROUTE, select
from tools.curriculum_riichi_policy_profile_v1 import profile as basic_profile
from tools.curriculum_calls_repair import prediction_index, OPERATIONS
from tools.curriculum_parent import require_replay

METHOD = 'calls13_basic_maxclass_rest_ce_v1'
MARKER = '_calls13_classroute_loss_v1'
STAGE = 'stage2-calls-classroute-v1'
CANDIDATE = 'registered_final_full_two_epochs_calls_classroute_v1'
COMPONENTS = (*BASIC_COMPONENTS, 'advanced_action_ce', 'post_reach_action_ce')


def key(src):
    if not src.startswith('curriculum2:') or src.removeprefix('curriculum2:') not in MASS:
        raise ValueError('Explicit thirteen-source loss category required')
    return src.removeprefix('curriculum2:')


def annotate(encoded, request):
    for q in encoded['questions']:
        raw = request['questions'][q['qid']]; source = key(raw['src'])
        if MARKER in q: raise ValueError('Do not annotate the same example twice')
        if source != POST:
            old_annotate({'questions': [q]}, request)
        else:
            if (q.get('target') is not None or raw.get('target') is not None or FIELD in q
                    or q['src'] != raw['src'] or len(q['keys']) != len(set(q['keys']))
                    or set(q['keys']) != set(raw['criteria']) or q['keys'][q['label']] != raw['label']
                    or {c['type'] for c in raw['criteria'].values()} != {'dahai'}):
                raise ValueError('Pristine aligned post-reach hard-target discards required')
            q[FIELD] = []
        q[MARKER] = METHOD
    return encoded


def parts(logits, question, device):
    source = key(question['src'])
    if question.get(MARKER) != METHOD: raise ValueError('Explicit new loss annotation required')
    if source != POST:
        result = old_parts(logits, question, device)
        return dict(result, post_reach_action_ce=result['objective']*0)
    import torch
    from torch.nn import functional as F
    label = question['label']
    if (question.get('target') is not None or question.get(FIELD) != []
            or logits.ndim != 1 or not torch.isfinite(logits).all()
            or type(label) is not int or not 0 <= label < len(logits)):
        raise ValueError('Finite aligned post-reach CE required')
    ce = F.cross_entropy(logits[None], torch.tensor([label], device=device)); zero = ce*0
    return {name: ce if name in ('objective', 'standard_action_ce', 'post_reach_action_ce') else zero
            for name in COMPONENTS}


def weighted_loss(logits, question, device, weights, observer=None):
    import torch
    source = key(question['src'])
    if set(weights) != set(MASS) or any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError('Complete registered positive thirteen-source weights required')
    result = parts(logits, question, device)
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=source, **{k: float(v.detach()) for k, v in result.items()}))
    return result['objective']*weights[source]


def summarize(pending, weights):
    if set(weights) != set(MASS) or any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError('Complete positive weights required')
    grouped = defaultdict(list)
    for row in pending:
        if row['source'] not in MASS or any(not math.isfinite(row[k]) for k in COMPONENTS):
            raise ValueError('Finite actual per-example loss components required')
        grouped[row['source']].append(row)
    if set(grouped) != set(MASS): raise ValueError('Actual update missing a source stratum')
    means = {s: {k: sum(r[k] for r in rows)/len(rows) for k in COMPONENTS} for s, rows in grouped.items()}
    counts = {s: len(rows) for s, rows in grouped.items()}; total = sum(counts.values())
    weighted = lambda k: sum(counts[s]*weights[s]*means[s][k] for s in means)/total
    return dict(source_loss_components=means, source_counts=counts,
                source_objective={s: m['objective'] for s, m in means.items()},
                source_ce={s: m['standard_action_ce'] for s, m in means.items()},
                loss=weighted('objective'), ce=weighted('standard_action_ce'))


def profile(records, predictions, mass, base, ce):
    if mass != MASS: raise ValueError('Registered thirteen-source loss mass required')
    saved = prediction_index(records, predictions); by = defaultdict(list)
    hits = defaultdict(list); raw_hits = defaultdict(list)
    ops = defaultdict(Counter); raw_ops = defaultdict(Counter)
    source_ops = defaultdict(lambda: defaultdict(Counter)); basic = []
    for row in records:
        q = row['questions']['action']; source = key(q['src']); p = saved[row['_meta']['id']]
        keys = p['keys']; types = [q['criteria'][k]['type'] for k in keys]
        chosen, _ = select(types, p['p'], list(range(len(keys))))
        raw = max(range(len(keys)), key=p['p'].__getitem__)
        hits[source].append(keys[chosen] == q['label']); raw_hits[source].append(keys[raw] == q['label'])
        truth = q['criteria'][q['label']]['type']
        for operation in set(types) & OPERATIONS:
            outcome = lambda actual: 'tp' if truth == operation and actual == operation else 'fn' if truth == operation else 'fp' if actual == operation else 'tn'
            ops[operation][outcome(types[chosen])] += 1
            raw_ops[operation][outcome(types[raw])] += 1
            source_ops[source][operation][outcome(types[chosen])] += 1
        if source.startswith('basic:'):
            item = copy.deepcopy(row); item['questions']['action']['src'] = 'curriculum1:'+source.removeprefix('basic:')
            basic.append(item)
            components = dict(cached_parts(item['questions']['action'], keys, p['p'], ce),
                              advanced_action_ce=0., post_reach_action_ce=0.)
        else:
            value = ce(q, keys, p['p']); components = {k: 0. for k in COMPONENTS}
            components.update(objective=value, standard_action_ce=value)
            components['post_reach_action_ce' if source == POST else 'advanced_action_ce'] = value
        by[source].append(components)
    if set(by) != set(MASS): raise ValueError('All thirteen actual validation sources required')
    bp = [saved[r['_meta']['id']] for r in basic]
    natural = sum(ce(r['questions']['action'], p['keys'], p['p']) for r, p in zip(basic, bp))/len(basic)
    b = basic_profile(basic, bp, {'weighted_target_ce': natural}, ce)
    means = {s: {k: sum(r[k] for r in rows)/len(rows) for k in COMPONENTS} for s, rows in by.items()}
    counts = {s: len(v) for s, v in by.items()}
    convert = lambda values: {s: {k: c[k] for k in ('tp', 'fn', 'tn', 'fp')} for s, c in values.items()}
    return dict(base, phase_weighted_ce=base['weighted_target_ce'],
        weighted_target_ce=sum(mass[s]*means[s]['standard_action_ce'] for s in mass),
        weighted_class_objective=sum(mass[s]*means[s]['objective'] for s in mass),
        natural_target_ce=sum(counts[s]*means[s]['standard_action_ce'] for s in mass)/len(records),
        source_loss_components=means, source_objective={s: m['objective'] for s, m in means.items()},
        stratum_ce={s: m['standard_action_ce'] for s, m in means.items()},
        stratum_accuracy={s: sum(v)/len(v) for s, v in hits.items()},
        raw_stratum_accuracy={s: sum(v)/len(v) for s, v in raw_hits.items()}, stratum_counts=counts,
        basic_replay=b['selected_profile'], raw_basic_replay=b['raw_profile'],
        basic_replay_source_quality=b['selected_source_quality'], raw_basic_source_quality=b['raw_source_quality'],
        riichi_type_counts=b['selected_profile']['riichi_type_counts'],
        riichi_by_source=b['selected_profile']['riichi_by_source'],
        operation_type_counts=convert(ops), raw_operation_type_counts=convert(raw_ops),
        operation_type_counts_by_source={s: convert(v) for s, v in source_ops.items()},
        loss_method=METHOD, selection_method=ROUTE,
        cached_agreement_all_candidates_not_native_guard_replay=True,
        action_agreement_is_not_policy_strength=True)


def warmstart_parity(parent_rows, parent_predictions, child_rows, child_predictions):
    old = prediction_index(parent_rows, parent_predictions); current = prediction_index(child_rows, child_predictions)
    require_replay(parent_rows, child_rows); error = 0.
    for row in parent_rows:
        identifier = row['_meta']['id']; a = old[identifier]; b = current[identifier]
        if a['keys'] != b['keys']: raise ValueError('Warm-start candidate order changed')
        error = max(error, max(abs(x-y) for x, y in zip(a['p'], b['p'])))
        types = [row['questions']['action']['criteria'][k]['type'] for k in a['keys']]
        if select(types, a['p'], list(range(len(types)))) != select(types, b['p'], list(range(len(types)))):
            raise ValueError('Warm-start fixed-selector choice changed')
    if len(old) != 3266 or error >= 5e-5: raise ValueError('Full parent probability/choice parity required')
    return dict(checked_basic_menus=len(old), max_probability_error=error,
                same_selected_choices=True, tolerance=5e-5)


def require_recipe(c, data):
    expected = layout(data['categories']['train'])
    values = dict(stage=STAGE, calls_loss_method=METHOD, selection_method=ROUTE,
        epochs=2, batch=1, accumulation=253, seed=2026112209, stage_offset=608,
        minimum_tail_fraction=.75, optimizer_resume=False, new_optimizer=True,
        all_thirteen_strata_each_update=True, each_epoch_without_replacement=True,
        same_adapter_and_pointer_head=True, learning_rate=1e-5, head_learning_rate=5e-5,
        validation_every_steps=194, game_evaluation_every_steps=194, candidate_rule=CANDIDATE,
        automatic_stage3_launch=False, automatic_release=False)
    if (any(c.get(k) != v for k, v in values.items()) or data['status'] != READY
            or data['loss_mass'] != MASS or c.get('loss_mass') != MASS
            or data['example_weights'] != expected['example_weights']
            or data['count_only_accumulation'] != expected['accumulation']
            or expected['final_step'] != 388 or expected['record_exposures'] != 98044
            or data['files']['validation']['records'] != 12479):
        raise ValueError('Prospective complete388-update class-route calls recipe changed')
    return expected
