"""Native one-action counterfactual data for both Jev Play policy heads."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import torch
from riichienv import GameRule, RiichiEnv
from jevmahjong.core.io import request_hash
from jevmahjong.core.seeds import engine_seed
from jevmahjong.core.state import build_request, legal_candidates
from jevmahjong.core.scoring import settlement_ledger
from jevmahjong.policy.routing import HybridPolicy, load_model
from jevmahjong.evaluation.agents import EfficiencyAgent, CallingOpponent, PolicyAgent, guarded_indices
from .rewards import alternate_indices, credit_pair, hand_outcomes, opportunity, scope_for

SUITES = ('closed', 'mixed', 'strong')


def action_key(action):
    return json.dumps(action.to_dict(), sort_keys=True)


def derived_seed(config_seed, seed, seat, replica, domain):
    material = f'{config_seed}:{seed}:{seat}:{replica}:{domain}'.encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], 'little')


def call_observation_digest(observations):
    """Native JSON serialization handles Meld objects in opened hands."""
    visible = {p: json.loads(base64.b64decode(obs.serialize_to_base64())) for p, obs in observations.items()}
    return hashlib.sha256(json.dumps(visible, sort_keys=True).encode()).hexdigest()


def replay_call_prefix(env, prefix, expected_digest):
    observations = env.reset()
    for batch in prefix:
        if env.done() or set(batch) != set(observations): raise ValueError('Call prefix player coverage differs')
        selected = {}
        for player, saved in batch.items():
            legal = {action_key(a): a for a in observations[player].legal_actions()}
            if action_key(saved) not in legal: raise ValueError('Call prefix contains an illegal action')
            selected[player] = legal[action_key(saved)]
        observations = env.step(selected)
    if env.done() or call_observation_digest(observations) != expected_digest:
        raise ValueError('Call intervention state differs from control')
    return observations


class CaptureClient:
    def __init__(self, router):
        self.router = router
        self.request = self.response = None

    def system_one(self, **request):
        self.request = request
        self.response = self.router.system_one(**request)
        return self.response


def suite_for_seed(seed, config):
    offset = seed - config['training_seed_start']
    if not 0 <= offset < config['training_seeds_per_stage']:
        raise ValueError('Unregistered collection wall')
    return SUITES[offset % len(SUITES)]


class NativeCollector:
    def __init__(self, config, device='cuda'):
        self.config = dict(config)
        self.ordinary = load_model(config['discard_checkpoint'], device)
        self.client = load_model(config['call_checkpoint'], device)
        self.model = self.client.runtime.model
        self.model.requires_grad_(False)
        self.ordinary.runtime.model.requires_grad_(False)
        self.router = HybridPolicy(self.ordinary, self.client)
        self.logged = CaptureClient(self.router)
        self.agent = PolicyAgent(self.logged)
        self.strong = PolicyAgent(self.ordinary)
        self.efficiency, self.caller = EfficiencyAgent(), CallingOpponent()
        self.controls = {}; self.checks = 0
        self.fast_inputs = []; self.capture_active = False; self.hooks = []
        for scope, head in [('ordinary', self.ordinary.runtime.model.head), ('calling', self.model.head)]:
            self.hooks.append(head.register_forward_pre_hook(
                lambda module, inputs, scope=scope: self.fast_inputs.append(
                    (scope, tuple(t.detach().clone().float() for t in inputs))) if self.capture_active else None))

    def _env(self, seed):
        return RiichiEnv(game_mode=self.config['game_mode'], rule=GameRule.default_mjsoul(),
            seed=engine_seed(seed, self.config['seed_scheme']))

    def _row(self, env, seed, seat, old_opponent, replica, trace_count):
        ledger = settlement_ledger(env.mjai_log, env.scores())
        return {'seed': seed, 'engine_seed': engine_seed(seed, self.config['seed_scheme']),
            'seat': seat, 'replica': replica, 'rank': int(env.ranks()[seat]), 'score': int(env.scores()[seat]),
            'suite': suite_for_seed(seed, self.config), 'trainable_decisions': trace_count,
            'point_components': ledger['components'][seat], 'all_point_components': ledger['components'],
            'settlement_ledger_passed': True, 'mjai_log': copy.deepcopy(env.mjai_log)}

    def act(self, obs, frozen=False, stochastic=False, trace=None):
        if not frozen or stochastic:
            raise ValueError('Frozen greedy continuation required')
        self.fast_inputs.clear(); self.logged.request = self.logged.response = None
        self.capture_active = trace is not None
        try:
            chosen = self.agent.act(obs)
        finally:
            self.capture_active = False
        request = self.logged.request
        scope = scope_for(request) if request is not None else None
        if trace is not None and scope:
            from jevany.api import validate_distribution
            if len(self.fast_inputs) != 1 or self.fast_inputs[0][0] != scope:
                raise ValueError('Wrong hybrid representation capture')
            decide, options = self.fast_inputs[0][1]
            head = self.ordinary.runtime.model.head if scope == 'ordinary' else self.model.head
            p = head(decide, options).softmax(-1).detach().cpu()
            keys = list(request['questions']['action']['criteria'])
            native, _ = validate_distribution(self.logged.response['answers']['action']['probabilities'], keys)
            if not torch.allclose(p, torch.tensor(native), atol=2e-5, rtol=2e-4):
                raise ValueError('Native head features fail probability replay')
            _, actions, features = build_request(obs)
            eligible = sorted(set(guarded_indices(obs, actions, features)) |
                {i for i, value in enumerate(features) if not value})
            teacher = next(i for i, a in enumerate(actions) if action_key(a) == action_key(chosen))
            if teacher != max(eligible, key=lambda i: float(p[i])):
                raise ValueError('Teacher differs from actual guarded action')
            if len(eligible) > 1:
                trace.append({'scope': scope, 'decide': decide.cpu(), 'options': options.cpu(),
                    'eligible': eligible, 'teacher_action': teacher, 'reference': p,
                    'action_keys': [action_key(a) for a in actions],
                    'public_request': copy.deepcopy(request), 'request_sha256': request_hash(request)})
                self.checks += 1
        self.fast_inputs.clear()
        return chosen

    def _collect_control(self, seed, seat, old_opponent):
        suite = suite_for_seed(seed, self.config)
        env = self._env(seed)
        observations, prefix, decisions = env.reset(), [], []
        while not env.done():
            selected = {}
            for player, obs in observations.items():
                if player == seat:
                    trace = []; selected[player] = self.act(obs, frozen=True, trace=trace)
                    if trace:
                        hands = hand_outcomes(env.mjai_log)
                        if len(trace) != 1 or not hands or hands[-1]['complete']:
                            raise ValueError('Decision does not belong to a current hand')
                        trace[0].update(hand_index=len(hands) - 1, draws_before=hands[-1]['draws'])
                        decisions.append({'step': len(prefix), 'digest': call_observation_digest(observations),
                            'row': trace[0], 'action': selected[player]})
                else:
                    selected[player] = self.opponent(player, obs, seat, suite)
            prefix.append(selected); observations = env.step(selected)
            if len(prefix) > 100000:
                raise RuntimeError('Control did not terminate')
        row = self._row(env, seed, seat, suite == 'strong', -1, 0)
        row.update(environment_steps=len(prefix), available_intervention_decisions=len(decisions), frozen_control=True)
        self.controls[seed, seat] = {'game': row, 'prefix': prefix, 'decisions': decisions}
        return dict(row), []

    def _intervene(self, seed, seat, old_opponent, replica):
        control = self.controls[seed, seat]
        category = self.config['intervention_categories'][replica]
        pool = [d for d in control['decisions'] if opportunity(d['row'], category)]
        if not pool:
            result = copy.deepcopy(control['game'])
            result.update(replica=replica, frozen_control=False, requested_category=category,
                trainable_decisions=0, reused_control_outcome=True, terminal_point_delta=0)
            if replica == self.config['training_replicas'] - 1:
                del self.controls[seed, seat]
            return result, []
        g = torch.Generator().manual_seed(derived_seed(self.config['seed'], seed, seat, replica, 'fast-position'))
        decision = pool[int(torch.randint(len(pool), (1,), generator=g))]
        row = copy.deepcopy(decision['row'])
        alternatives = alternate_indices(row, category)
        g.manual_seed(derived_seed(self.config['seed'], seed, seat, replica, 'fast-action'))
        index = alternatives[int(torch.randint(len(alternatives), (1,), generator=g))]
        env = self._env(seed)
        observations = replay_call_prefix(env, control['prefix'][:decision['step']], decision['digest'])
        actions = legal_candidates(observations[seat])
        if [action_key(a) for a in actions] != row['action_keys']:
            raise ValueError('Intervention legal actions changed after exact replay')
        hands = hand_outcomes(env.mjai_log)
        if len(hands) - 1 != row['hand_index'] or hands[-1]['draws'] != row['draws_before']:
            raise ValueError('Intervention hand or timing changed after replay')
        selected = {}
        for player, obs in observations.items():
            saved = actions[index] if player == seat else control['prefix'][decision['step']][player]
            selected[player] = next(a for a in obs.legal_actions() if action_key(a) == action_key(saved))
        observations = env.step(selected); steps = decision['step'] + 1
        suite = suite_for_seed(seed, self.config)
        while not env.done():
            selected = {p: self.act(o, frozen=True) if p == seat else self.opponent(p, o, seat, suite)
                for p, o in observations.items()}
            observations = env.step(selected); steps += 1
            if steps > 100000:
                raise RuntimeError('Branch did not terminate')
        result = self._row(env, seed, seat, suite == 'strong', replica, 1)
        credit = credit_pair(control['game']['mjai_log'], result['mjai_log'], row['hand_index'],
            seat, row['draws_before'], self.config['fast_gamma'])
        row.update(action=index, episode_key=(seed, seat, replica), category=category,
            alternative_probability=1. / len(alternatives), **credit)
        result.update(requested_category=category, reused_control_outcome=False,
            intervention_step=decision['step'], intervention_state_digest=decision['digest'],
            environment_steps=steps, terminal_point_delta=result['score'] - control['game']['score'], **credit)
        if replica == self.config['training_replicas'] - 1:
            del self.controls[seed, seat]
        return result, [row]

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.controls.clear()

    def opponent(self, player, obs, seat, suite):
        if suite not in SUITES or player == seat:
            raise ValueError('Invalid opponent routing')
        policy = self.efficiency
        if player == (seat + 1) % 4:
            if suite == 'mixed': policy = self.caller
            if suite == 'strong': policy = self.strong
        return policy.act(obs)

    def game(self, seed, seat, training=False, frozen=False, replica=0):
        if training and not frozen:
            return self._intervene(seed, seat, False, replica)
        if frozen and not training and replica == -1:
            return self._collect_control(seed, seat, False)
        raise ValueError('Collector accepts only controls and one-action branches')
