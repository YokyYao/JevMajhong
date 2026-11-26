"""Grade the actual fixed RI class choice without rewriting raw probabilities.

Cached-menu agreement uses all listed candidates, as the prior validation
contract does. It is not a replay of the native discard guard. RI class choice
is independent of which discard that guard retains; native strength still
needs the separately audited policy games. No parent is admitted here.
"""
from collections import Counter, defaultdict
import copy

from tools.curriculum_basic_repair import MASS, grade, profile as raw_profile
from tools.curriculum_calls_repair import prediction_index
from tools.curriculum_riichi_class_policy_v1 import METHOD, select


def profile(records, predictions, base, ce):
    saved = prediction_index(records, predictions)
    raw = raw_profile(records, predictions, MASS, base, ce)
    selected = copy.deepcopy(raw)
    source_hits = defaultdict(list); category_hits = defaultdict(list)
    confusion = {source: Counter() for source in ('legacy', 'teacher')}
    scopes = Counter(); changes = Counter()
    for row in records:
        question = row['questions']['action']
        source_tag = question['src']
        if not source_tag.startswith('curriculum1:'):
            raise ValueError('Explicit six-source basic projection required')
        key = source_tag.removeprefix('curriculum1:')
        if key not in MASS:
            raise ValueError('Unknown basic source stratum')
        source, category = key.split(':')
        prediction = saved[row['_meta']['id']]
        keys = prediction['keys']; probabilities = prediction['p']
        types = [question['criteria'][k]['type'] for k in keys]
        chosen, applied = select(types, probabilities, list(range(len(keys))))
        baseline = max(range(len(keys)), key=probabilities.__getitem__)
        if applied != (category != 'plain'):
            raise ValueError('Basic source tag differs from actual fixed-RI menu scope')
        if category != 'plain' and (types[keys.index(question['label'])] == 'reach') != (category == 'riichi_yes'):
            raise ValueError('RI target tag differs from actual source label')
        hit = keys[chosen] == question['label']
        source_hits[key].append(hit); category_hits[category].append(hit)
        scopes['RI' if applied else 'non_RI'] += 1
        changes['RI' if applied else 'non_RI'] += chosen != baseline
        if applied:
            yes = category == 'riichi_yes'; actual = types[chosen] == 'reach'
            confusion[source]['tp' if yes and actual else 'fn' if yes else 'fp' if actual else 'tn'] += 1
    mean = lambda grouped: {key: sum(values)/len(values) for key, values in grouped.items()}
    selected.update(source_stratum_accuracy=mean(source_hits), stratum_accuracy=mean(category_hits),
                    riichi_by_source={s: dict(c) for s, c in confusion.items()},
                    riichi_type_counts=dict(confusion['legacy']+confusion['teacher']))
    return dict(method=METHOD, raw_profile=raw, selected_profile=selected,
                raw_source_quality=grade(raw), selected_source_quality=grade(selected),
                menus=len(records), scope_counts=dict(scopes), changed_choices=dict(changes),
                original_probabilities_and_cross_entropy_unchanged=True,
                cached_menu_all_candidates_agreement_not_native_guard_replay=True,
                native_strength_or_parent_admission_proven=False)
