"""Validation arithmetic for mixed replay and conditional future targets."""
from collections import Counter, defaultdict
import math

from tools.curriculum_attack_losses import canonical_question


def target_ce(question, keys, probabilities):
    _, q = canonical_question(question)
    if len(keys) != len(probabilities) or len(set(keys)) != len(keys) or set(keys) != set(q['criteria']):
        raise ValueError('Prediction menu differs')
    if any(not math.isfinite(v) or v < 0 for v in probabilities) or abs(sum(probabilities)-1) > 1e-5:
        raise ValueError('Invalid prediction distribution')
    values = dict(zip(keys, probabilities))
    if q.get('target') is None:
        return -math.log(max(values[q['label']], 1e-30)), 0.
    target = q['target']
    if set(target)-set(keys) or any(not math.isfinite(v) or v < 0 for v in target.values()):
        raise ValueError('Invalid stored target')
    selected = {k: v for k, v in target.items() if v > 0}
    if len(selected) < 2 or abs(sum(selected.values())-1) > 1e-6:
        raise ValueError('An informative normalized MC target is required')
    total = sum(selected.values())
    mass = sum(values[k] for k in selected)
    if mass <= 0:
        raise ValueError('No prediction mass on labelled support')
    ce = -sum(v/total*math.log(max(values[k]/mass, 1e-30)) for k, v in selected.items())
    entropy = -sum(v/total*math.log(v/total) for v in selected.values())
    return ce, entropy


def profile(records, predictions, phase_weights, loss_mass):
    byid = {r['_meta']['id']: r for r in records}
    if len(byid) != len(records) or len(predictions) != len(records) or {p['id'] for p in predictions} != set(byid):
        raise ValueError('Validation dropped, repeated or changed rows')
    ces, entropies, phases, hits, expert_hits = [defaultdict(list) for _ in range(5)]
    riichi = Counter()
    operations = defaultdict(Counter)
    unknown = Counter()
    for p in predictions:
        row = byid[p['id']]
        q = row['questions']['action']
        kind, _ = canonical_question(q)
        ce, entropy = target_ce(q, p['keys'], p['p'])
        ces[kind].append(ce)
        entropies[kind].append(entropy)
        phase = row['_meta']['phase']
        phases[phase].append(ce)
        chosen = p['keys'][max(range(len(p['p'])), key=p['p'].__getitem__)]
        hits[kind].append(chosen == q['label'])
        if kind.startswith('mc:'):
            unknown[kind] += q['target'].get(chosen, 0.) <= 0
            continue
        expert_hits[phase].append(chosen == q['label'])
        actual = q['criteria'][chosen]['type']
        truth = q['criteria'][q['label']]['type']
        if kind in ('expert:riichi_yes', 'expert:riichi_no'):
            yes = kind == 'expert:riichi_yes'
            selected = actual == 'reach'
            riichi['tp' if yes and selected else 'fn' if yes else 'fp' if selected else 'tn'] += 1
        for op in {v['type'] for v in q['criteria'].values()} & {'chi', 'pon', 'ankan', 'daiminkan', 'kakan', 'reach'}:
            yes, selected = truth == op, actual == op
            operations[op]['tp' if yes and selected else 'fn' if yes else 'fp' if selected else 'tn'] += 1
    if set(ces) != set(loss_mass) or set(phases) != set(phase_weights):
        raise ValueError('Validation source or phase coverage differs')
    if any(not math.isfinite(v) or v <= 0 for v in [*loss_mass.values(), *phase_weights.values()]) or abs(sum(loss_mass.values())-1) > 1e-12:
        raise ValueError('Registered validation masses are invalid')
    means = {k: sum(v)/len(v) for k, v in ces.items()}
    entropy_means = {k: sum(v)/len(v) for k, v in entropies.items()}
    phase_means = {k: sum(v)/len(v) for k, v in phases.items()}
    weighted = sum(loss_mass[k]*means[k] for k in loss_mass)
    entropy = sum(loss_mass[k]*entropy_means[k] for k in loss_mass)
    return dict(weighted_target_ce=weighted, weighted_target_KL=weighted-entropy,
        weighted_target_entropy=entropy, natural_target_ce=sum(sum(v) for v in ces.values())/len(records),
        phase_weighted_ce=sum(phase_weights[k]*phase_means[k] for k in phase_weights)/sum(phase_weights.values()),
        phase_target_ce=phase_means, phase_records={k: len(v) for k, v in phases.items()},
        expert_accuracy={k: sum(v)/len(v) for k, v in expert_hits.items()},
        stratum_ce=means, stratum_target_entropy=entropy_means,
        stratum_accuracy={k: sum(v)/len(v) for k, v in hits.items()}, stratum_counts={k: len(v) for k, v in ces.items()},
        riichi_type_counts={k: riichi[k] for k in ('tp', 'fn', 'tn', 'fp')},
        operation_type_counts={k: {c: v[c] for c in ('tp', 'fn', 'tn', 'fp')} for k, v in operations.items()},
        MC_unlabelled_actions_selected=dict(unknown),
        MC_argmax_agreement_is_not_expert_accuracy=True, pseudo_labels_are_not_expert_truth=True)
