"""Progression training metrics and source-complete mixing-block epoch RNG."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

from tools.curriculum_progression_objective_v1 import MASS, METHOD, COMPONENTS, annotate, parts, epoch_order
from tools.curriculum_complete_legal_class_policy_v1 import select, METHOD as ROUTE
from tools.curriculum_calls_classroute_v1 import profile as calls_profile
from tools.curriculum_calls_post_reach_data_v1 import MASS as CALLS_MASS


def weighted_loss(logits, question, device, weights, observer=None):
    import torch
    if set(weights) != set(MASS) or any(not math.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError('All nineteen registered positive source weights required')
    result = parts(logits, question, device)
    if observer is not None and torch.is_grad_enabled() and logits.requires_grad:
        observer.append(dict(source=question['src'], **{k: float(v.detach()) for k, v in result.items()}))
    return result['objective']*weights[question['src']]


def summarize(pending, weights):
    if (not pending or set(weights) != set(MASS)
            or any(not math.isfinite(v) or v <= 0 for v in weights.values())):
        raise ValueError('Actual optimizer examples and registered positive weights required')
    groups = defaultdict(list)
    for row in pending:
        if row['source'] not in MASS or any(not math.isfinite(row[k]) for k in COMPONENTS):
            raise ValueError('Actual finite loss components required')
        groups[row['source']].append(row)
    means = {s: {k: sum(r[k] for r in rows)/len(rows) for k in COMPONENTS} for s, rows in groups.items()}
    counts = {s: len(rows) for s, rows in groups.items()}; n = sum(counts.values())
    weighted = lambda field: sum(counts[s]*weights[s]*means[s][field] for s in means)/n
    return dict(source_loss_components=means, source_counts=counts,
        source_objective={s: m['objective'] for s, m in means.items()},
        source_ce={s: m['standard_target_ce'] for s, m in means.items()},
        loss=weighted('objective'), ce=weighted('standard_target_ce'),
        extra_RI_class_loss=weighted('extra_RI_group_ce'))


class ProgressionEpochRandom(random.Random):
    def __init__(self, seed, categories, directory, expected):
        super().__init__(seed); self.categories = list(categories); self.directory = Path(directory)
        self.expected = expected; self.epochs = 0

    def shuffle(self, order):
        if sorted(order) != list(range(87595)): raise ValueError('Exact original full TRAIN index set required')
        rng = random.Random(); rng.setstate(self.getstate())
        selected, receipt = epoch_order(self.categories, rng); self.setstate(rng.getstate()); self.epochs += 1
        digest = hashlib.sha256(json.dumps(selected).encode()).hexdigest()
        if self.epochs > 2 or digest != self.expected[self.epochs-1]['order_sha256']:
            raise ValueError('Actual epoch differs from its preflighted full permutation')
        order[:] = selected; self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory/f'epoch-{self.epochs:02d}.json').open('x') as stream:
            json.dump(dict(epoch=self.epochs, order_sha256=digest, **receipt), stream, indent=2); stream.write('\n')


def random_proxy(original, seed, categories, directory, expected):
    created = []
    def factory(value=None):
        if type(value) is int and value == seed:
            if created: raise ValueError('Unexpected duplicate main training RNG')
            rng = ProgressionEpochRandom(value, categories, directory, expected); created.append(rng); return rng
        return original.Random(value)
    return SimpleNamespace(seed=original.seed, Random=factory), created


def profile(records, predictions, base, old_ce):
    import torch
    from jevany.data import materialize
    byid = {r['_meta']['id']: r for r in records}; saved = {p['id']: p for p in predictions}
    if len(byid) != len(records) or len(saved) != len(predictions) or set(byid) != set(saved):
        raise ValueError('Predictions must cover every original distinct union row')
    sources = defaultdict(list); hits = defaultdict(list); mc_hits = defaultdict(list)
    calls = []; calls_predictions = []; ops = defaultdict(Counter); hard = 0
    for row in records:
        q = row['questions']['action']; src = q['src']; p = saved[row['_meta']['id']]; keys = p['keys']
        if (src not in MASS or len(set(keys)) != len(keys) or set(keys) != set(q['criteria'])
                or len(keys) != len(p['p']) or any(not math.isfinite(v) or v < 0 for v in p['p'])
                or abs(sum(p['p'])-1) > 1e-5): raise ValueError('Valid original source, candidate order and probability menu required')
        kinds = [q['criteria'][k]['type'] for k in keys]; chosen, _ = select(kinds, p['p'])
        encoded = annotate(materialize(row), row)['questions'][0]
        if encoded['keys'] != keys: raise ValueError('Cached validation menu reordered actual materializer')
        # Explicit floor for finite diagnostic reconstruction, normalized by
        # the same log-softmax as the numerical loss. No weights are loaded.
        logits = torch.tensor([math.log(max(v, 1e-30)) for v in p['p']], dtype=torch.float64)
        values = parts(logits, encoded, 'cpu'); sources[src].append({k: float(v) for k, v in values.items()})
        if q.get('target') is not None:
            mc_hits[src].append(q['target'].get(keys[chosen], 0.))
        else:
            hard += 1; hits[src].append(keys[chosen] == q['label']); truth = q['criteria'][q['label']]['type']; actual = kinds[chosen]
            for operation in set(kinds) & {'reach', 'chi', 'pon', 'ankan', 'daiminkan', 'kakan'}:
                outcome = 'tp' if truth == operation and actual == operation else 'fn' if truth == operation else 'fp' if actual == operation else 'tn'
                ops[operation][outcome] += 1
        if src.startswith('curriculum2:'):
            calls.append(row); calls_predictions.append(p)
    if set(sources) != set(MASS) or len(calls) != 12479: raise ValueError('All nineteen sources and complete old calls validation required')
    old = calls_profile(calls, calls_predictions, CALLS_MASS, {'weighted_target_ce': 0.}, old_ce)
    means = {s: {k: sum(r[k] for r in rows)/len(rows) for k in COMPONENTS} for s, rows in sources.items()}
    weighted = lambda field: sum(MASS[s]*means[s][field] for s in MASS)
    return dict(base, weighted_class_objective=weighted('objective'), weighted_target_ce=weighted('standard_target_ce'),
        natural_target_ce=sum(sum(r['standard_target_ce'] for r in rs) for rs in sources.values())/len(records),
        extra_RI_class_loss=weighted('extra_RI_group_ce'), source_loss_components=means,
        source_objective={s: m['objective'] for s, m in means.items()}, stratum_ce={s: m['standard_target_ce'] for s, m in means.items()},
        stratum_counts={s: len(rows) for s, rows in sources.items()},
        stratum_hard_accuracy={s: sum(v)/len(v) for s, v in hits.items()},
        MC_selected_mean_target_mass={s: sum(v)/len(v) for s, v in mc_hits.items()},
        MC_selected_target_mass_is_not_policy_value=True, hard_target_records=hard,
        basic_replay=old['basic_replay'], raw_basic_replay=old['raw_basic_replay'],
        basic_replay_source_quality=old['basic_replay_source_quality'], raw_basic_source_quality=old['raw_basic_source_quality'],
        riichi_type_counts=old['riichi_type_counts'], riichi_by_source=old['riichi_by_source'],
        operation_type_counts={s: {k: v[k] for k in ('tp', 'fn', 'tn', 'fp')} for s, v in ops.items()},
        original_calls_operation_type_counts=old['operation_type_counts'],
        original_calls_operation_type_counts_by_source=old['operation_type_counts_by_source'],
        original_calls_loss_diagnostic_uses_old_objective=True, loss_method=METHOD, selection_method=ROUTE,
        cached_hard_agreement_is_not_match_strength=True)
