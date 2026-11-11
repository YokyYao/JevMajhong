"""Stage2-native, one-action counterfactual East points with three opponent suites.

The policy and wall are frozen; only one legal chi/pon/pass action changes.
Hidden native observations are used only for exact replay verification, never
as model features. All per-game data remain private.
"""
import copy
import os
from pathlib import Path

import torch
from arena import EfficiencyAgent
from call_counterfactual import CaptureClient, call_observation_digest, replay_call_prefix
from call_policy import digest
from clean_policy_runtime import CleanPolicyRouter
from encoder_call_policy import EncoderCallRouter
from feature_policy_runtime import FeatureClient
from guide_call_collector import GuideCallInterventionPolicy
from scripts.evaluate_call_repair import Agent, CallingOpponent
from semantic import legal_candidates
from single_decision_rl import action_key, derived_seed

SUITES = ('closed', 'mixed', 'strong')


def suite_for_seed(seed, config):
    offset = seed - config['training_seed_start']
    if not 0 <= offset < config['training_seeds_per_stage']:
        raise ValueError('Unregistered collection wall')
    return SUITES[offset % len(SUITES)]


def alternate_action(count, teacher, generator):
    """Uniform over every legal alternative, excluding the known zero label."""
    if count < 2 or not 0 <= teacher < count:
        raise ValueError('Need a valid teacher and at least one alternative')
    draw = int(torch.randint(count - 1, (1,), generator=generator))
    return draw + (draw >= teacher)


class FeaturePointPolicy(GuideCallInterventionPolicy):
    def __init__(self, config):
        from jevany import JevModel
        from jevany.checkpoint import LoadOptions
        from jevany.inference import InferenceOptions
        self.config = dict(config)
        if (config['game_mode'] != '4p-red-east' or config['seed_scheme'] != 'spaced-v1'
                or config['reward_mode'] != 'net_points' or config['actor_updates'] != 0
                or config['intervention_sampling'] != 'uniform_non_teacher'):
            raise ValueError('Frozen East terminal-point collection required')
        if digest(Path(config['reference']) / 'head.pt') != config['reference_head_sha256']:
            raise ValueError('Ordinary reference changed')
        if digest(Path(config['teacher_checkpoint']) / 'head.pt') != config['teacher_head_sha256']:
            raise ValueError('Stage2 teacher changed')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        base = Path(os.environ.get('JEV_BASE_MODEL_DIR', Path.home() /
            '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))

        def load(path):
            return JevModel.from_pretrained(str(path), device='cuda', dtype='bf16',
                options=LoadOptions(merge=False, temperature=1., compile_mode=None,
                    cuda_graphs=False, fused_kernels=False, base_load_path=str(base)),
                inference_options=InferenceOptions(max_state_tokens=4096,
                    max_branch_tokens=8192, max_packed_tokens=8192, prefix_cache_size=0))

        self.ordinary = load(config['reference'])
        self.client = load(config['teacher_checkpoint'])
        self.model = self.client.runtime.model
        self.model.requires_grad_(False)
        self.ordinary.runtime.model.requires_grad_(False)
        self.base_head = self.reference = self.head = self.model.head
        self.capture = []
        self.capture_active = False
        self.hook = self.reference.register_forward_pre_hook(
            lambda module, inputs: self.capture.append(tuple(t.detach() for t in inputs))
            if self.capture_active else None)
        self.router = EncoderCallRouter(self.ordinary,
            CleanPolicyRouter(self.ordinary, FeatureClient(self.client)))
        self.logged = CaptureClient(self.router)
        self.agent = Agent(self.logged)
        self.strong = Agent(self.ordinary)
        self.efficiency, self.caller = EfficiencyAgent(), CallingOpponent()
        self.controls = {}
        self.checks = 0

    def opponent(self, player, obs, seat, suite):
        if suite not in SUITES or player == seat:
            raise ValueError('Invalid opponent routing')
        policy = self.efficiency
        if player == (seat + 1) % 4:
            if suite == 'mixed': policy = self.caller
            if suite == 'strong': policy = self.strong
        return policy.act(obs)

    def _row(self, env, seed, seat, old_opponent, replica, trace_count):
        row = super()._row(env, seed, seat, old_opponent, replica, trace_count)
        row.update(suite=suite_for_seed(seed, self.config),
            collection_method='stage2_one_alternative_frozen_continuation')
        return row

    def _collect_control(self, seed, seat, old_opponent):
        suite = suite_for_seed(seed, self.config)
        env = self._env(seed)
        observations, prefix, decisions = env.reset(), [], []
        while not env.done():
            selected = {}
            for player, obs in observations.items():
                if player == seat:
                    trace = []
                    selected[player] = self.act(obs, frozen=True, trace=trace)
                    if trace:
                        if len(trace) != 1: raise ValueError('Multiple call traces')
                        decisions.append({'step': len(prefix),
                            'digest': call_observation_digest(observations),
                            'row': trace[0], 'action': selected[player]})
                else:
                    selected[player] = self.opponent(player, obs, seat, suite)
            prefix.append(selected)
            observations = env.step(selected)
            if len(prefix) > 100000: raise RuntimeError('Control did not terminate')
        row = self._row(env, seed, seat, suite == 'strong', -1, 0)
        row.update(environment_steps=len(prefix), available_intervention_decisions=len(decisions),
            frozen_control=True)
        self.controls[seed, seat] = {'game': row, 'prefix': prefix, 'decisions': decisions}
        return dict(row), []

    def _intervene(self, seed, seat, old_opponent, replica):
        control = self.controls[seed, seat]
        suite = suite_for_seed(seed, self.config)
        if control['game']['suite'] != suite: raise ValueError('Opponent suite changed')
        position_seed = derived_seed(self.config['seed'], seed, seat, replica, 'position')
        sampling_seed = derived_seed(self.config['seed'], seed, seat, replica, 'action')
        result = copy.deepcopy(control['game'])
        result.update(replica=replica, frozen_control=False, position_seed=position_seed,
            sampling_seed=sampling_seed, reused_control_outcome=True, trainable_decisions=0,
            simulated_prefix_steps=0, simulated_continuation_steps=0)
        rows = []
        decisions = control['decisions']
        if decisions:
            position = int(torch.randint(len(decisions), (1,),
                generator=torch.Generator().manual_seed(position_seed)))
            decision = decisions[position]
            row = copy.deepcopy(decision['row'])
            teacher = row['action']
            count = len(row['options'])
            index = alternate_action(count, teacher, torch.Generator().manual_seed(sampling_seed))
            p = torch.full((count,), 1. / (count - 1))
            p[teacher] = 0.
            row.update(action=index, old_logp=float(p[index].log()), behavior_probabilities=p,
                intervention_decision=position, intervention_step=decision['step'],
                intervention_state_digest=decision['digest'], teacher_action=teacher,
                intervention_sampling='uniform_non_teacher')
            env = self._env(seed)
            observations = replay_call_prefix(env, control['prefix'][:decision['step']], decision['digest'])
            actions = legal_candidates(observations[seat])
            if ([action_key(a) for a in actions] != row['action_keys']
                    or action_key(actions[teacher]) != action_key(decision['action'])):
                raise ValueError('Replayed intervention actions differ')
            selected = {}
            for player, obs in observations.items():
                if player == seat: selected[player] = actions[index]
                else:
                    saved = control['prefix'][decision['step']][player]
                    selected[player] = next(a for a in obs.legal_actions() if action_key(a) == action_key(saved))
            observations = env.step(selected)
            steps = decision['step'] + 1
            while not env.done():
                selected = {p: self.act(o, frozen=True) if p == seat
                    else self.opponent(p, o, seat, suite) for p, o in observations.items()}
                observations = env.step(selected)
                steps += 1
                if steps > 100000: raise RuntimeError('Intervention did not terminate')
            outcome = self._row(env, seed, seat, suite == 'strong', replica, 1)
            for field in ('rank', 'score', 'reward', 'point_components', 'settlement_ledger_passed'):
                result[field] = copy.deepcopy(outcome[field])
            result.update(trainable_decisions=1, intervention_decision=position,
                intervention_step=decision['step'], intervention_state_digest=decision['digest'],
                teacher_action=teacher, sampled_action=index, environment_steps=steps,
                reused_control_outcome=False, simulated_prefix_steps=decision['step'],
                simulated_continuation_steps=steps - decision['step'])
            row.update(episode_key=(seed, seat, replica), **{'return': result['reward']})
            rows = [row]
        result['control_reward'] = control['game']['reward']
        result['terminal_point_delta'] = result['score'] - control['game']['score']
        if replica == self.config['training_replicas'] - 1: del self.controls[seed, seat]
        return result, rows

    def game(self, seed, seat, training=False, frozen=False, old_opponent=False, replica=0):
        if old_opponent: raise ValueError('Use the registered cyclic opponent suite')
        if training and not frozen: return self._intervene(seed, seat, False, replica)
        if frozen and not training and replica == -1: return self._collect_control(seed, seat, False)
        raise ValueError('Collection only; use independent native games for strength')
