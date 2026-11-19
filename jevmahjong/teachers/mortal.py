"""Local Mortal teacher, using only the acting player's event stream.

Upstream code/weights are external dependencies, not redistributed here.
See docs/experiments/plans/mortal-teacher-20261119.md for provenance.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

WEIGHT_SHA256 = "bfb3a6c072aa0bfd4171a9cdc77cb6c02ae42cde920843f9e5784394f23447d8"
BINARY_SHA256 = "3aeb09112198f6babe9bba4dd392ea009669ee7ba5d33cd3cd4242c737bb004b"


def public_events(events, seat):
    """Reject accidental oracle input before invoking the native teacher."""
    result = []
    for raw in events:
        e = json.loads(raw) if isinstance(raw, str) else dict(raw)
        if e['type'] == 'start_game':
            e.update(id=seat, names=['0', '1', '2', '3'])
        if e['type'] == 'start_kyoku':
            if any(t != '?' for p, hand in enumerate(e['tehais']) if p != seat for t in hand):
                raise ValueError('Teacher received an opponent concealed hand')
        if e['type'] == 'tsumo' and e['actor'] != seat and e['pai'] != '?':
            raise ValueError('Teacher received an opponent concealed draw')
        result.append(e)
    return result


def action_key(action):
    value = json.loads(action.to_mjai())
    # Match the native legal_candidates canonicalization, including red tiles.
    return json.dumps({k: sorted(v) if k == 'consumed' else v
                       for k, v in value.items() if k not in ('actor', 'target', 'tsumogiri')}, sort_keys=True)


def legal_index(action, candidates):
    matches = [i for i, a in enumerate(candidates) if action_key(a) == action_key(action)]
    if len(matches) != 1:
        raise ValueError('Teacher action does not map uniquely into the native legal menu')
    return matches[0]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MortalTeacher:
    def __init__(self, root):
        if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
                or socket.gethostname().startswith(('lo-', 'login'))):
            raise RuntimeError('Mortal loading requires a scheduler GPU allocation')
        import torch
        root = Path(root)
        weights = root / 'mortal-298k.pth'
        binary = root / 'mortal-adapter-reference/libriichi/libriichi-3.12-x86_64-unknown-linux-gnu.so'
        if hashlib.sha256(weights.read_bytes()).hexdigest() != WEIGHT_SHA256:
            raise ValueError('Unexpected Mortal weights')
        if hashlib.sha256(binary.read_bytes()).hexdigest() != BINARY_SHA256:
            raise ValueError('Unexpected libriichi extension')
        loader = _module('_jev_mortal_loader', root / 'mortal-adapter-reference/_libriichi_loader.py')
        loader.load()
        import libriichi
        if Path(libriichi.__file__).resolve() != binary.resolve():
            raise ValueError('Another libriichi module was already loaded')
        model = _module('_jev_mortal_model', root / 'mortal-upstream/mortal/model.py')
        engine = _module('_jev_mortal_engine', root / 'mortal-upstream/mortal/engine.py')
        saved = torch.load(weights, map_location='cpu', weights_only=True)
        cfg = saved['config']
        if (cfg['control']['version'] != 4 or cfg['resnet']['conv_channels'] != 192
                or cfg['resnet']['num_blocks'] != 40):
            raise ValueError('Unexpected teacher architecture')
        brain = model.Brain(version=4, conv_channels=192, num_blocks=40)
        dqn = model.DQN(version=4)
        brain.load_state_dict(saved['mortal'], strict=True)
        dqn.load_state_dict(saved['current_dqn'], strict=True)
        self.engine = engine.MortalEngine(brain, dqn, False, 4, device=torch.device('cuda'),
            enable_amp=False, enable_rule_based_agari_guard=False, name='VoidShine-Mortal-298k')
        from libriichi.mjai import Bot
        self.Bot = Bot
        self.config = cfg

    def player(self, seat):
        return MortalSeat(self, seat)


class MortalSeat:
    def __init__(self, teacher, seat):
        self.teacher = teacher
        self.seat = seat
        self.bot = teacher.Bot(teacher.engine, seat)
        self.last_reaction = None
        self.decisions = 0

    def act(self, obs):
        from jevmahjong.core.state import legal_candidates
        if obs.player_id != self.seat:
            raise ValueError('Teacher seat mismatch')
        events = public_events(obs.events, self.seat)
        response = None
        for i, e in enumerate(events):
            response = self.bot.react(json.dumps(e), can_act=i == len(events)-1)
        candidates = legal_candidates(obs)
        if len(candidates) == 1:
            self.last_reaction = None
            return candidates[0]
        if response is None:
            raise ValueError('Teacher produced no legal decision; no rule fallback is allowed')
        self.last_reaction = json.loads(response)
        decision = {k: v for k, v in self.last_reaction.items() if k != 'meta'}
        action = obs.select_action_from_mjai(decision)
        if action is None:
            raise ValueError('Cannot translate Mortal decision into native action')
        index = legal_index(action, candidates)
        self.decisions += 1
        return candidates[index]
