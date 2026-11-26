"""Loss-only strata for attack/risk training, with explicit masked semantics."""
from collections import Counter
import math

from tools.build_curriculum_stage2 import category
from tools.curriculum_losses import question_loss

PREFIX = 'curriculum3:'
EXPERT = {'plain', 'riichi_yes', 'riichi_no', 'open', 'call', 'kan'}
MC = {'closed_discard', 'open_discard', 'call_window', 'kan_window'}


def stratum(row):
    label = row['_meta']['label_type']
    if label == 'expert_replay':
        if row['questions']['action'].get('target') is not None:
            raise ValueError('Expert replay must retain its observed hard label')
        return 'expert:'+category(row)
    if label == 'paired_future_masked' and row['_meta']['phase'] in MC:
        target = row['questions']['action'].get('target')
        if not isinstance(target, dict) or sum(v > 0 for v in target.values()) < 2:
            raise ValueError('Informative masked future target required')
        return 'mc:'+row['_meta']['phase']
    raise ValueError('Unexpected source/phase')


def mixture(rows, expert_mass=.75):
    if not math.isfinite(expert_mass) or not 0 < expert_mass < 1:
        raise ValueError('Both replay and simulation must have positive mass')
    counts = Counter(stratum(row) for row in rows)
    expected = {'expert:'+k for k in EXPERT} | {'mc:'+k for k in MC}
    if set(counts) != expected:
        raise ValueError('Every replay and MC stratum must be covered')
    riichi = counts['expert:riichi_yes']+counts['expert:riichi_no']
    # Preserve the calls course's relative replay mixture, including the
    # natural accept/decline ratio, rather than letting ordinary examples
    # dominate again when simulated cases are appended.
    mass = {'expert:'+k: expert_mass*v for k, v in dict(plain=.25, open=.25, call=.25, kan=.15,
        riichi_yes=.1*counts['expert:riichi_yes']/riichi,
        riichi_no=.1*counts['expert:riichi_no']/riichi).items()}
    mass.update({'mc:'+k: (1-expert_mass)/len(MC) for k in MC})
    weights = {k: mass[k]*len(rows)/counts[k] for k in counts}
    if abs(sum(mass.values())-1) > 1e-12 or max(weights.values()) > 8:
        raise ValueError('Invalid loss mass or excessive rare-source weight')
    return dict(counts=dict(counts), loss_mass=mass, example_weights=weights)


def canonical_question(question):
    src = question.get('src', '')
    if not isinstance(src, str) or not src.startswith(PREFIX):
        raise ValueError('Explicit attack-course loss tag required')
    key = src[len(PREFIX):]
    if key in {'mc:'+p for p in MC}:
        if question.get('target') is None:
            raise ValueError('MC target was lost')
        canonical = 'paired_future_masked'
    elif key in {'expert:'+p for p in EXPERT}:
        if question.get('target') is not None:
            raise ValueError('Expert label unexpectedly became a soft target')
        canonical = 'expert_replay'
    else:
        raise ValueError('Unregistered attack-course loss stratum')
    return key, dict(question, src=canonical)


def loss(logits, question, device, weights):
    key, restored = canonical_question(question)
    if key not in weights:
        raise ValueError('Loss weight is not registered')
    return question_loss(logits, restored, device)*weights[key]
