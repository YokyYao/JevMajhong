"""Actual retained-parent objective profiles; unchanged policy diagnostics."""
from collections import defaultdict
import math

from tools.curriculum_progression_parent_loss_v1 import METHOD, MASS, COMPONENTS, annotate, parts
from tools.curriculum_progression_paired_v1 import random_proxy
from tools.curriculum_progression_fit_v1 import profile as previous_profile


def profile(records, predictions, base, old_ce):
    import torch
    from jevany.data import materialize
    # Preserve the source confusion matrices and all original grading semantics.
    result = previous_profile(records, predictions, base, old_ce)
    saved = {p['id']: p for p in predictions}
    sources = defaultdict(list)
    for row in records:
        pred = saved[row['_meta']['id']]
        q = annotate(materialize(row), row)['questions'][0]
        if q['keys'] != pred['keys']:
            raise ValueError('Actual validation menu order changed')
        logits = torch.tensor([math.log(max(v, 1e-30)) for v in pred['p']], dtype=torch.float64)
        sources[q['src']].append({k: float(v) for k, v in parts(logits, q, 'cpu').items()})
    means = {s: {k: sum(r[k] for r in rows)/len(rows) for k in COMPONENTS}
             for s, rows in sources.items()}
    weighted = lambda k: sum(MASS[s]*means[s][k] for s in MASS)
    if (set(means) != set(MASS)
            or abs(weighted('standard_target_ce')-result['weighted_target_ce']) > 1e-10):
        raise ValueError('Source-complete canonical target CE must remain identical')
    result.pop('extra_RI_class_loss')
    result.update(weighted_class_objective=weighted('objective'),
        signed_objective_adjustment=weighted('signed_objective_minus_standard_ce'),
        source_loss_components=means,
        source_objective={s: m['objective'] for s, m in means.items()},
        stratum_ce={s: m['standard_target_ce'] for s, m in means.items()},
        loss_method=METHOD)
    if abs(result['weighted_class_objective']-result['weighted_target_ce']
           -result['signed_objective_adjustment']) > 1e-10:
        raise ValueError('Signed actual validation objective accounting differs')
    return result
