"""Prospective yaku-course source metrics and exact epoch RNG adapter."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

from tools.curriculum_yaku_objective_v1 import MASS, METHOD, MARKER, COMPONENTS, NEW_SOURCES, RECORDS, annotate, parts, epoch_order
from tools.curriculum_progression_fit_v1 import profile as replay_profile
from tools.curriculum_progression_objective_v1 import MASS as REPLAY_MASS
from tools.curriculum_complete_legal_class_policy_v1 import select, METHOD as ROUTE


class YakuEpochRandom(random.Random):
    def __init__(self, seed, categories, directory, expected):
        super().__init__(seed)
        self.categories = list(categories); self.directory = Path(directory)
        self.expected = expected; self.epochs = 0

    def shuffle(self, order):
        if sorted(order) != list(range(RECORDS)):
            raise ValueError('Exact complete yaku TRAIN index set required')
        rng = random.Random(); rng.setstate(self.getstate())
        selected, receipt = epoch_order(self.categories, rng)
        self.setstate(rng.getstate()); self.epochs += 1
        digest = hashlib.sha256(json.dumps(selected).encode()).hexdigest()
        if self.epochs > 2 or digest != self.expected[self.epochs-1]['order_sha256']:
            raise ValueError('Actual yaku epoch differs from its prepared full order')
        order[:] = selected; self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory/f'epoch-{self.epochs:02d}.json').open('x') as stream:
            json.dump(dict(epoch=self.epochs, order_sha256=digest, **receipt), stream, indent=2); stream.write('\n')


def random_proxy(original, seed, categories, directory, expected):
    created = []
    def factory(value=None):
        if type(value) is int and value == seed:
            if created:
                raise ValueError('Unexpected duplicate main yaku training RNG')
            result = YakuEpochRandom(seed, categories, directory, expected); created.append(result); return result
        return original.Random(value)
    return SimpleNamespace(seed=original.seed, Random=factory), created


def profile(records, predictions, base, old_ce):
    import torch
    from jevany.data import materialize
    rows = {r['_meta']['id']: r for r in records}; saved = {p['id']: p for p in predictions}
    if (len(rows) != 24516 or len(saved) != len(predictions) or len(rows) != len(records)
            or set(rows) != set(saved)):
        raise ValueError('Complete24516 actual distinct validation predictions required')
    replay = [r for r in records if r['questions']['action']['src'] in REPLAY_MASS]
    if len(replay) != 23688:
        raise ValueError('Unchanged full progression validation replay required')
    inherited = replay_profile(replay, [saved[r['_meta']['id']] for r in replay], {}, old_ce)
    means = dict(inherited['source_loss_components']); source_counts = dict(inherited['stratum_counts'])
    new_values = defaultdict(list); new_hits = defaultdict(list); new_ops = defaultdict(Counter)
    phases = defaultdict(list)
    for row in records:
        q = row['questions']['action']; src = q['src']
        if src in REPLAY_MASS:
            continue
        p = saved[row['_meta']['id']]; keys = p['keys']
        if (src not in NEW_SOURCES or q.get('target') is not None or len(keys) != len(set(keys))
                or set(keys) != set(q['criteria']) or len(keys) != len(p['p'])
                or any(not math.isfinite(v) or v < 0 for v in p['p']) or abs(sum(p['p'])-1) > 1e-5):
            raise ValueError('Valid actual new-source hard-label probability menu required')
        material = annotate(materialize(row), row)['questions'][0]
        if material['keys'] != keys:
            raise ValueError('Actual new validation menu reordered')
        values = parts(torch.tensor([math.log(max(v, 1e-30)) for v in p['p']], dtype=torch.float64), material, 'cpu')
        new_values[src].append({k: float(v) for k, v in values.items()})
        kinds = [q['criteria'][k]['type'] for k in keys]; chosen, _ = select(kinds, p['p'])
        hit = keys[chosen] == q['label']; new_hits[src].append(hit)
        phases[row['_meta']['phase']].append(hit)
        truth = q['criteria'][q['label']]['type']; actual = kinds[chosen]
        for op in set(kinds) & {'reach', 'chi', 'pon', 'ankan', 'daiminkan', 'kakan', 'ryukyoku'}:
            name = 'tp' if truth == op and actual == op else 'fn' if truth == op else 'fp' if actual == op else 'tn'
            new_ops[op][name] += 1
    if set(new_values) != set(NEW_SOURCES):
        raise ValueError('All six actual new source pools required')
    for src, values in new_values.items():
        means[src] = {k: sum(v[k] for v in values)/len(values) for k in COMPONENTS}
        source_counts[src] = len(values)
    if set(means) != set(MASS):
        raise ValueError('Complete25-source validation profile required')
    weighted = lambda name: sum(MASS[s]*means[s][name] for s in MASS)
    result = dict(base, weighted_class_objective=weighted('objective'), weighted_target_ce=weighted('standard_target_ce'),
        natural_target_ce=sum(source_counts[s]*means[s]['standard_target_ce'] for s in MASS)/len(records),
        extra_RI_class_loss=weighted('extra_RI_group_ce'), source_loss_components=means,
        source_objective={s: v['objective'] for s, v in means.items()},
        stratum_ce={s: v['standard_target_ce'] for s, v in means.items()}, stratum_counts=source_counts,
        new_source_teacher_agreement={s: sum(v)/len(v) for s, v in new_hits.items()},
        stratum_hard_accuracy=dict(inherited['stratum_hard_accuracy'],
            **{s: sum(v)/len(v) for s, v in new_hits.items()}),
        hard_target_records=inherited['hard_target_records']+sum(len(v) for v in new_hits.values()),
        MC_selected_mean_target_mass=inherited['MC_selected_mean_target_mass'],
        MC_selected_target_mass_is_not_policy_value=True,
        new_source_teacher_agreement_is_not_yaku_or_game_mastery=True,
        new_phase_teacher_agreement={s: dict(records=len(v), agreement=sum(v)/len(v)) for s, v in phases.items()},
        new_operation_type_counts={s: {k: v[k] for k in ('tp', 'fn', 'tn', 'fp')} for s, v in new_ops.items()},
        replay_profile=inherited, loss_method=METHOD, selection_method=ROUTE,
        cached_hard_agreement_is_not_match_strength=True)
    # Preserve the exact original retention surface, kept separate from the new
    # source agreement. Additional RI/yaku rows cannot dilute or hide failures.
    for key in ('basic_replay', 'raw_basic_replay', 'basic_replay_source_quality',
                'raw_basic_source_quality', 'riichi_type_counts', 'riichi_by_source',
                'original_calls_operation_type_counts', 'original_calls_operation_type_counts_by_source',
                'original_calls_loss_diagnostic_uses_old_objective'):
        result[key] = inherited[key]
    return result
