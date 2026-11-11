"""Reuse a completed reference arm only when its gameplay contract matches."""
import json
from pathlib import Path
from jevmahjong.core.io import digest
from jevmahjong.training.provenance import verify
from .metrics import summarize

GAMEPLAY_FILES = ('core/state.py', 'core/seeds.py', 'core/scoring.py', 'core/io.py',
    'policy/routing.py', 'policy/call_features.py', 'policy/call_helpers.py',
    'data/policy_data.py', 'evaluation/agents.py', 'evaluation/game.py')


def gameplay_identity(protocol):
    files = {p.split('/jevmahjong/', 1)[-1]: h for p, h in protocol['code_sha256'].items()}
    return {p: files[p] for p in GAMEPLAY_FILES}


def load_reference(directory, config, protocol):
    directory = Path(directory)
    previous = json.loads((directory / 'protocol.json').read_text())
    audit = json.loads((directory / 'audit.json').read_text())
    saved = json.loads((directory / 'result.json').read_text())
    raw = directory / 'games-private.jsonl'
    if audit['status'] != 'passed_complete' or digest(raw) != audit['games_sha256']:
        raise ValueError('Complete unchanged reference source required')
    verify(previous)
    if (any(previous['config'][k] != config[k] for k in ('stage', 'seed_start', 'seeds', 'fast_gamma'))
            or previous['model_files'] != protocol['model_files']
            or gameplay_identity(previous) != gameplay_identity(protocol)):
        raise ValueError('Reference gameplay contract differs')
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    actual = summarize(rows, previous['config'])
    if (audit['games'] != len(rows) or actual['games'] != saved['games']
            or actual['overall']['candidate_minus_reference'] != saved['overall']['candidate_minus_reference']
            or actual['fast_reward_paired'] != saved['fast_reward_paired']):
        raise ValueError('Reference source point/speed reaggregation differs')
    return [r for r in rows if r['policy'] == 'reference'], {
        'source': str(directory.resolve()), 'games_sha256': digest(raw),
        'protocol_sha256': digest(directory / 'protocol.json')}
