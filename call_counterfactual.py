"""Native chi/pon-only Q_frozen pairs; discard behavior remains the incumbent."""
import copy
import base64
import hashlib
import json
from pathlib import Path

import torch
from call_policy import CallRouter, digest, is_call_request, request_hash
from semantic import legal_candidates
from single_decision_rl import SingleDecisionPolicy, action_key, derived_seed
from arena import EfficiencyAgent
from scripts.evaluate_call_repair import Agent
from scripts.train_call_repair import make_head


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


class CallInterventionPolicy(SingleDecisionPolicy):
    """Reuse validated prefix replay/credit; trace only actual native call choices.

    Three EfficiencyAgent opponents during collection. Evaluation must still use
    the separately frozen arena protocol. No parameter updates occur here.
    """
    def __init__(self, config):
        from jevany import JevModel
        from jevany.checkpoint import LoadOptions
        from jevany.inference import InferenceOptions
        self.config = dict(config)
        if config['game_mode'] != '4p-red-east' or config['seed_scheme'] != 'spaced-v1':
            raise ValueError('Point-call pilot requires the fixed East independent-wall scheme')
        if config['reward_mode'] != 'net_points' or config['actor_updates'] != 0:
            raise ValueError('Collect frozen terminal-point labels without actor updates')
        reference, checkpoint = Path(config['reference']), Path(config['call_checkpoint'])
        if digest(reference / 'head.pt') != config['reference_head_sha256'] or digest(checkpoint) != config['call_head_sha256']:
            raise ValueError('Frozen base/call identity mismatch')
        meta = torch.load(reference / 'head.pt', map_location='cpu', weights_only=True)
        call = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if call['reference_head_sha256'] != config['reference_head_sha256']: raise ValueError('Call head source differs')
        if set(call['scope']) != {'chi', 'pon', 'none'}: raise ValueError('Call-only head scope required')
        if any(not torch.equal(call['head'][k], meta['head'][k]) for k in ('k.weight', 'k.bias')):
            raise ValueError('Call key projection differs from base')
        base = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'
        self.client = JevModel.from_pretrained(str(reference), device='cuda', dtype='bf16',
            options=LoadOptions(merge=False, temperature=1., compile_mode=None, cuda_graphs=False,
                fused_kernels=False, base_load_path=str(base)),
            inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192,
                                              max_packed_tokens=8192, prefix_cache_size=0))
        self.model = self.client.runtime.model; self.model.requires_grad_(False)
        self.base_head = self.model.head
        self.reference = make_head(meta, 'cuda').eval().requires_grad_(False)
        self.reference.load_state_dict(call['head']); self.head = self.reference
        self.capture = []; self.capture_active = False
        self.hook = self.reference.register_forward_pre_hook(
            lambda module, inputs: self.capture.append(tuple(t.detach() for t in inputs)) if self.capture_active else None)
        self.logged = CaptureClient(CallRouter(self.client, self.reference))
        self.agent = Agent(self.logged); self.controls = {}; self.checks = 0

    def act(self, obs, frozen=False, stochastic=False, trace=None):
        if not frozen or stochastic: raise ValueError('Control and continuation must be frozen greedy policies')
        from jevany.api import validate_distribution
        self.capture.clear(); self.logged.request = self.logged.response = None
        self.capture_active = True
        try: chosen = self.agent.act(obs)
        finally: self.capture_active = False
        if self.model.head is not self.base_head: raise ValueError('Call router failed to restore discard head')
        request = self.logged.request
        if trace is not None and request is not None and is_call_request(request):
            if len(self.capture) != 1: raise ValueError('Missing native call pointer capture')
            decide, options = (t.clone().to(dtype=torch.float32) for t in self.capture.pop())
            actions = legal_candidates(obs); keys = list(request['questions']['action']['criteria'])
            probabilities, _ = validate_distribution(self.logged.response['answers']['action']['probabilities'], keys)
            with torch.no_grad(): replayed = self.reference(decide, options).softmax(-1)
            if len(actions) != len(options) or not torch.allclose(torch.tensor(probabilities, device=replayed.device), replayed, atol=2e-5, rtol=2e-4):
                raise ValueError('Native call feature/probability alignment failed')
            index = next(i for i, a in enumerate(actions) if action_key(a) == action_key(chosen))
            if int(replayed.argmax()) != index: raise ValueError('Cached teacher greedy action mismatch')
            self.checks += 1
            trace.append({'decide': decide.cpu(), 'options': options.cpu(), 'eligible': list(range(len(actions))),
                'action': index, 'old_logp': float(replayed[index].log()), 'reference': replayed.cpu(), 'old_value': 0.,
                'types': [a['type'] for a in request['questions']['action']['criteria'].values()],
                'action_keys': [action_key(a) for a in actions],
                'public_request': copy.deepcopy(request), 'request_sha256': request_hash(request),
                'scope': 'native_chi_pon_none_only'})
        self.capture.clear()
        return chosen

    # Keep the historical generic collector unchanged. These call-specific
    # methods use native observation serialization for already-opened hands.
    def _collect_control(self, seed, seat, old_opponent):
        env, baseline = self._env(seed), EfficiencyAgent()
        observations, prefix, decisions = env.reset(), [], []
        while not env.done():
            selected = {}
            for player, obs in observations.items():
                if player == seat:
                    trace = []; selected[player] = self.act(obs, frozen=True, trace=trace)
                    if trace:
                        if len(trace) != 1: raise ValueError('Exactly one call trace per decision')
                        decisions.append({'step': len(prefix), 'digest': call_observation_digest(observations),
                                          'row': trace[0], 'action': selected[player]})
                else: selected[player] = baseline.act(obs)
            prefix.append(selected); observations = env.step(selected)
            if len(prefix) > 100000: raise RuntimeError('Call control failed to terminate')
        row = self._row(env, seed, seat, False, -1, 0)
        row.update(environment_steps=len(prefix), available_intervention_decisions=len(decisions), frozen_control=True)
        self.controls[seed, seat] = {'game': row, 'prefix': prefix, 'decisions': decisions}
        return dict(row), []

    def _intervene(self, seed, seat, old_opponent, replica):
        from head_rl import mixed_probabilities
        control = self.controls[seed, seat]
        if control['game']['old_opponent']: raise ValueError('Call collection opponent mismatch')
        position_seed = derived_seed(self.config['seed'], seed, seat, replica, 'position')
        sampling_seed = derived_seed(self.config['seed'], seed, seat, replica, 'action')
        position_rng = torch.Generator().manual_seed(position_seed)
        action_rng = torch.Generator().manual_seed(sampling_seed)
        result = dict(control['game'])
        result.update(replica=replica, frozen_control=False, position_seed=position_seed,
            sampling_seed=sampling_seed, reused_control_outcome=True, trainable_decisions=0,
            simulated_prefix_steps=0, simulated_continuation_steps=0)
        rows = []; decisions = control['decisions']
        if decisions:
            position = int(torch.randint(len(decisions), (1,), generator=position_rng)); decision = decisions[position]
            row = dict(decision['row']); device = next(self.reference.parameters()).device
            with torch.no_grad():
                p = mixed_probabilities(self.reference(row['decide'].to(device), row['options'].to(device)),
                                        row['eligible'], self.config['exploration_uniform']).cpu()
            index = int(torch.multinomial(p, 1, generator=action_rng)); teacher_index = row['action']
            row.update(action=index, old_logp=float(p[index].log()), reference=p, old_value=0.,
                intervention_decision=position, intervention_step=decision['step'],
                intervention_state_digest=decision['digest'], teacher_action=teacher_index)
            result.update(trainable_decisions=1, intervention_decision=position,
                intervention_step=decision['step'], intervention_state_digest=decision['digest'],
                teacher_action=teacher_index, sampled_action=index)
            if index != teacher_index:
                env = self._env(seed)
                observations = replay_call_prefix(env, control['prefix'][:decision['step']], decision['digest'])
                actions = legal_candidates(observations[seat])
                if (len(actions) != len(row['options'])
                    or action_key(actions[row['eligible'][teacher_index]]) != action_key(decision['action'])
                    or [action_key(a) for a in actions] != row['action_keys']):
                    raise ValueError('Call action/feature alignment differs after replay')
                selected = {}
                for player, obs in observations.items():
                    if player == seat: selected[player] = actions[row['eligible'][index]]
                    else:
                        saved = control['prefix'][decision['step']][player]
                        selected[player] = next(a for a in obs.legal_actions() if action_key(a) == action_key(saved))
                observations = env.step(selected); steps = decision['step'] + 1; baseline = EfficiencyAgent()
                while not env.done():
                    selected = {p: self.act(o, frozen=True) if p == seat else baseline.act(o) for p, o in observations.items()}
                    observations = env.step(selected); steps += 1
                    if steps > 100000: raise RuntimeError('Call branch failed to terminate')
                outcome = self._row(env, seed, seat, False, replica, 1)
                for field in ('rank', 'score', 'reward'): result[field] = outcome[field]
                result.update(environment_steps=steps, reused_control_outcome=False,
                    simulated_prefix_steps=decision['step'], simulated_continuation_steps=steps - decision['step'])
            row.update(episode_key=(seed, seat, replica), **{'return': result['reward']}); rows = [row]
        result['control_reward'] = control['game']['reward']
        result['terminal_point_delta'] = result['score'] - control['game']['score']
        if replica == self.config['training_replicas'] - 1: del self.controls[seed, seat]
        return result, rows

    def game(self, seed, seat, training=False, frozen=False, old_opponent=False, replica=0):
        if old_opponent: raise ValueError('Call collection uses three fixed EfficiencyAgent opponents')
        if training: return self._intervene(seed, seat, False, replica)
        if frozen and replica == -1: return self._collect_control(seed, seat, False)
        raise ValueError('This collector is not a strength evaluator; use the native arena')

    def close(self):
        self.hook.remove()
