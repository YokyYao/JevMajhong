"""Actor-free replay of forced-win source logs, never expert action labels.

MJAI omits suppressed call intentions. Canonical PASS for those actors is
allowed only after the resulting native event prefix exactly matches the
recorded source. Every offered legal win is still taken. No model is loaded.
"""


def exact_prefix(actual, expected):
    return len(actual) <= len(expected) and actual == expected[:len(actual)]


def resolve_batch(env, observations, expected):
    from riichienv import Phase, ActionType
    from jevmahjong.core.state import legal_candidates
    if env.done() or not exact_prefix(env.mjai_log, expected):
        raise ValueError('Live native source prefix required')
    before = len(env.mjai_log)
    menus = {p: legal_candidates(obs) for p, obs in sorted(observations.items())}
    if any(not menu for menu in menus.values()):
        raise ValueError('An active actor has no legal candidate')
    wins = {p: [a for a in menu if a.action_type in (ActionType.RON, ActionType.TSUMO)]
            for p, menu in menus.items()}
    if env.phase == Phase.WaitAct:
        if len(menus) != 1: raise ValueError('Exactly one active draw actor required')
        seat = next(iter(menus))
        choices = [{seat: a} for a in (wins[seat] or menus[seat])]
    elif env.phase == Phase.WaitResponse:
        baseline = {}
        for p, menu in menus.items():
            passes = [a for a in menu if a.action_type == ActionType.PASS]
            if wins[p]: baseline[p] = wins[p][0]
            elif len(passes) == 1: baseline[p] = passes[0]
            else: raise ValueError('Explicit non-winning PASS required in a response window')
        choices = [baseline]
        # A simultaneous winning batch takes precedence. If nobody can win,
        # only one call can execute; unknown losing intentions stay canonical
        # PASS and must never be exported as negative teacher labels.
        if not any(wins.values()):
            for p, menu in menus.items():
                choices.extend([{**baseline, p: a} for a in menu if a.action_type != ActionType.PASS])
    else:
        raise ValueError('Unverified native phase in actor-free source replay')
    for trial, selected in enumerate(choices, 1):
        fork = env.clone(); fresh = fork.step(selected)
        if len(fork.mjai_log) > before and exact_prefix(fork.mjai_log, expected):
            return fork, fresh, selected, dict(native_trials=trial,
                legal_wins_taken=sum(bool(v) for v in wins.values()),
                canonical_missing_intentions_not_expert_labels=True)
    raise ValueError('No forced-win canonical batch reproduces the next complete native event prefix')


def replay(env, observations, expected, *, before_step=None, max_steps=30000):
    if type(max_steps) is not int or not 1 <= max_steps <= 30000:
        raise ValueError('Bounded replay steps required')
    trials = wins = 0
    for step in range(max_steps):
        if env.done():
            if env.mjai_log != expected: raise ValueError('Terminal source event stream differs')
            return env, dict(native_batches=step, native_trials=trials, forced_wins=wins)
        if before_step is not None: before_step(env, observations, step)
        env, observations, selected, receipt = resolve_batch(env, observations, expected)
        trials += receipt['native_trials']; wins += receipt['legal_wins_taken']
    raise RuntimeError('Bounded source replay did not finish')
