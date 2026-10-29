"""Frozen-policy, one-decision counterfactual returns for a head-only pilot.

This optimizes a sampled Q_teacher surrogate, not a full-trajectory on-policy PPO
objective. Only one sampled decision receives the paired terminal-point delta.
"""
import hashlib
import json

import torch
from riichienv import GameRule, RiichiEnv

from arena import EfficiencyAgent
from head_rl import HeadPolicy, mixed_probabilities, terminal_reward
from seed_control import engine_seed
from semantic import legal_candidates


def derived_seed(config_seed, seed, seat, replica, domain):
    material = f'{config_seed}:{seed}:{seat}:{replica}:{domain}'.encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], 'little')


def observation_digest(observations):
    visible = {p: obs.to_dict() for p, obs in observations.items()}
    return hashlib.sha256(json.dumps(visible, sort_keys=True).encode()).hexdigest()


def action_key(action):
    return json.dumps(action.to_dict(), sort_keys=True)


def replay_prefix(env, prefix, expected_digest):
    """Use fresh legal Action objects, and require the exact intervention state."""
    observations = env.reset()
    for batch in prefix:
        if env.done() or set(batch) != set(observations):
            raise ValueError('Counterfactual prefix player coverage differs')
        selected = {}
        for player, saved in batch.items():
            legal = {action_key(a): a for a in observations[player].legal_actions()}
            if action_key(saved) not in legal:
                raise ValueError('Counterfactual prefix contains an illegal action')
            selected[player] = legal[action_key(saved)]
        observations = env.step(selected)
    if env.done() or observation_digest(observations) != expected_digest:
        raise ValueError('Counterfactual intervention state differs from its control')
    return observations


class SingleDecisionPolicy(HeadPolicy):
    def __init__(self, config, checkpoint=None):
        if checkpoint is not None or config['max_stages'] != 1:
            raise ValueError('Single-decision pilot needs one fresh collection stage')
        if config['advantage_estimator'] != 'greedy_reference' or config['reward_mode'] != 'net_points':
            raise ValueError('Single-decision pilot requires paired terminal net points')
        super().__init__(config)
        if any(not torch.equal(v, self.reference.state_dict()[k]) for k, v in self.head.state_dict().items()):
            raise ValueError('Behavior head must equal the frozen teacher at collection')
        self.controls = {}

    def _env(self, seed):
        return RiichiEnv(game_mode=self.config['game_mode'], rule=GameRule.default_mjsoul(),
                         seed=engine_seed(seed, self.config['seed_scheme']))

    def _row(self, env, seed, seat, old_opponent, replica, trace_count):
        rank, score = int(env.ranks()[seat]), int(env.scores()[seat])
        return {'seed': seed, 'engine_seed': engine_seed(seed, self.config['seed_scheme']),
                'seed_scheme': self.config['seed_scheme'], 'seat': seat, 'rank': rank, 'score': score,
                'game_mode': self.config['game_mode'], 'reward_mode': 'net_points',
                'reward_scale': self.config['reward_scale'],
                'reward': terminal_reward(rank, 'net_points', score, self.config['reward_scale']),
                'trainable_decisions': trace_count, 'replica': replica, 'sampling_seed': None,
                'old_opponent': old_opponent, 'collection_method': 'one_sampled_decision_frozen_continuation'}

    def _collect_control(self, seed, seat, old_opponent):
        env, baseline = self._env(seed), EfficiencyAgent()
        observations, prefix, decisions = env.reset(), [], []
        while not env.done():
            selected = {}
            for player, obs in observations.items():
                if player == seat:
                    trace = []
                    selected[player] = self.act(obs, frozen=True, trace=trace)
                    if trace:
                        decisions.append({'step': len(prefix), 'digest': observation_digest(observations),
                                          'row': trace[0], 'action': selected[player]})
                elif old_opponent and player == (seat + 1) % 4:
                    selected[player] = self.act(obs, frozen=True)
                else:
                    selected[player] = baseline.act(obs)
            prefix.append(selected)
            observations = env.step(selected)
            if len(prefix) > 100000:
                raise RuntimeError('Control failed to terminate')
        row = self._row(env, seed, seat, old_opponent, -1, 0)
        row.update(environment_steps=len(prefix), available_intervention_decisions=len(decisions),
                   frozen_control=True)
        self.controls[seed, seat] = {'game': row, 'prefix': prefix, 'decisions': decisions}
        return dict(row), []

    def _intervene(self, seed, seat, old_opponent, replica):
        control = self.controls[seed, seat]
        if control['game']['old_opponent'] != old_opponent:
            raise ValueError('Control and intervention opponents differ')
        position_seed = derived_seed(self.config['seed'], seed, seat, replica, 'position')
        sampling_seed = derived_seed(self.config['seed'], seed, seat, replica, 'action')
        position_rng = torch.Generator().manual_seed(position_seed)
        action_rng = torch.Generator().manual_seed(sampling_seed)
        decisions = control['decisions']
        result = dict(control['game'])
        result.update(replica=replica, frozen_control=False, position_seed=position_seed,
                      sampling_seed=sampling_seed, reused_control_outcome=True, trainable_decisions=0,
                      simulated_prefix_steps=0, simulated_continuation_steps=0)
        rows = []
        if decisions:
            position = int(torch.randint(len(decisions), (1,), generator=position_rng))
            decision = decisions[position]
            row = dict(decision['row'])
            with torch.no_grad():
                device = next(self.reference.parameters()).device
                logits = self.reference(row['decide'].to(device), row['options'].to(device))
                probabilities = mixed_probabilities(logits, row['eligible'], self.config['exploration_uniform']).cpu()
            index = int(torch.multinomial(probabilities, 1, generator=action_rng))
            teacher_index = row['action']
            row.update(action=index, old_logp=float(probabilities[index].log()), reference=probabilities,
                       old_value=0.0, intervention_decision=position, intervention_step=decision['step'],
                       intervention_state_digest=decision['digest'], teacher_action=teacher_index)
            result.update(trainable_decisions=1, intervention_decision=position,
                          intervention_step=decision['step'], intervention_state_digest=decision['digest'],
                          teacher_action=teacher_index, sampled_action=index)
            # Same state, same action, same deterministic continuation: exact same outcome.
            if index != teacher_index:
                env = self._env(seed)
                observations = replay_prefix(env, control['prefix'][:decision['step']], decision['digest'])
                actions = legal_candidates(observations[seat])
                if len(actions) != len(row['options']) or action_key(actions[row['eligible'][teacher_index]]) != action_key(decision['action']):
                    raise ValueError('Counterfactual cached action alignment differs')
                selected = {}
                baseline = EfficiencyAgent()
                steps = decision['step']
                for player, obs in observations.items():
                    if player == seat:
                        selected[player] = actions[row['eligible'][index]]
                    else:
                        # Simultaneous opponents cannot observe the replacement before env.step.
                        saved = control['prefix'][decision['step']][player]
                        selected[player] = next(a for a in obs.legal_actions() if action_key(a) == action_key(saved))
                observations = env.step(selected)
                steps += 1
                while not env.done():
                    selected = {p: self.act(o, frozen=True) if p == seat or (old_opponent and p == (seat+1) % 4)
                                else baseline.act(o) for p, o in observations.items()}
                    observations = env.step(selected)
                    steps += 1
                    if steps > 100000:
                        raise RuntimeError('Counterfactual failed to terminate')
                outcome = self._row(env, seed, seat, old_opponent, replica, 1)
                for field in ('rank', 'score', 'reward'):
                    result[field] = outcome[field]
                result.update(environment_steps=steps, reused_control_outcome=False,
                              simulated_prefix_steps=decision['step'],
                              simulated_continuation_steps=steps-decision['step'])
            row.update(episode_key=(seed, seat, replica), **{'return': result['reward']})
            rows = [row]
        result['control_reward'] = control['game']['reward']
        result['terminal_point_delta'] = result['score'] - control['game']['score']
        if replica == self.config['training_replicas'] - 1:
            del self.controls[seed, seat]
        return result, rows

    def game(self, seed, seat, training=False, frozen=False, old_opponent=False, replica=0):
        if training:
            return self._intervene(seed, seat, old_opponent, replica)
        if frozen and replica == -1:
            return self._collect_control(seed, seat, old_opponent)
        # Evaluation always uses the full native greedy policy; never counterfactual reuse.
        return super().game(seed, seat, training, frozen, old_opponent, replica)
