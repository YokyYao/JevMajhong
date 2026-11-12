"""Short paired East-match evaluation for a frozen Jev Tempo candidate."""
import argparse
import fcntl
import json
from pathlib import Path
import torch
from jevmahjong.core.io import digest, write
from jevmahjong.policy.routing import HybridPolicy, HeadPolicyClient, load_model, make_head
from jevmahjong.policy.kan import KanPolicyClient, is_kan_request
from jevmahjong.policy.kan_context import KanContextPolicyClient, CallContextPolicyClient, probabilities as context_probabilities
from jevmahjong.training.provenance import snapshot, verify
from .agents import PolicyAgent
from .game import play
from .metrics import summarize, SUITES, POLICIES
from .reference_cache import load_reference


def candidate_scopes(config):
    scopes = config.get('updated_scopes', ['ordinary', 'calling'])
    if (not isinstance(scopes, list) or not scopes or len(set(scopes)) != len(scopes)
            or not set(scopes) <= {'ordinary', 'calling'}):
        raise ValueError('Unique nonempty ordinary/calling updated scopes required')
    return tuple(s for s in ('ordinary', 'calling') if s in scopes)


def select_heads(heads, saved, config):
    for scope in candidate_scopes(config):
        heads[scope].load_state_dict(saved[scope])


def candidate_activation(config, trained_filter='all'):
    mode = config.get('candidate_activation', 'all')
    if mode not in ('all', 'kan_windows', 'call_windows'):
        raise ValueError('Unknown candidate activation')
    if mode == 'kan_windows' and candidate_scopes(config) != ('ordinary',):
        raise ValueError('Kan activation requires only the ordinary updated head')
    if mode == 'call_windows' and candidate_scopes(config) != ('calling',):
        raise ValueError('Call activation requires only the calling scope')
    if trained_filter in ('kan_windows','call_windows') and mode != trained_filter:
        raise ValueError('Context training requires the registered activation')
    return mode


def parity_sources(fit):
    """EV-only datasets retain original native inputs for unchanged routes."""
    paths = {}
    for source in fit.get('data_sources', []):
        path = Path(source['path']) / 'protocol.json'
        if digest(path) != source['protocol_sha256']:
            raise ValueError('Data provenance changed')
        protocol = json.loads(path.read_text())
        if protocol.get('collection_method') == 'paired_future_ev_v1':
            for original,h in protocol['source_files'].items():
                if original in paths and paths[original] != h:raise ValueError('Conflicting parity source identity')
                paths[original] = h
    for path, h in fit['data_sha256'].items():
        if path in paths and paths[path] != h:raise ValueError('Conflicting parity source identity')
        paths[path] = h
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text()); out = Path(cfg['out'])
    if not isinstance(cfg.get('preflight_only',False),bool):raise ValueError('Boolean preflight flag required')
    scopes = candidate_scopes(cfg)
    activation = candidate_activation(cfg)
    fit_run = Path(cfg['candidate_run']); fit = json.loads((fit_run / 'result.json').read_text())
    candidate_activation(cfg, fit.get('decision_filter', fit.get('config', {}).get('decision_filter', 'all')))
    if fit['status'] != 'trained': raise ValueError('A completed trained candidate is required')
    verify(json.loads((fit_run / 'protocol.json').read_text()))
    checkpoint = Path(fit['selected']['checkpoint'])
    if digest(checkpoint) != fit['selected']['sha256']: raise ValueError('Selected candidate changed')
    out.mkdir(parents=True, exist_ok=args.resume)
    lock=(out/'evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if cfg['stage'] == 'independent_confirmation':
        development = Path(cfg['development_out'])
        previous = json.loads((development / 'result.json').read_text())
        audit = json.loads((development / 'audit.json').read_text())
        development_protocol = json.loads((development / 'protocol.json').read_text())
        verify(development_protocol)
        if candidate_scopes(development_protocol['config']) != scopes:
            raise ValueError('Confirmation must preserve development head scopes')
        if candidate_activation(development_protocol['config']) != activation:
            raise ValueError('Confirmation must preserve development activation')
        if (audit['checkpoint_sha256'] != digest(checkpoint)
                or digest(development / 'games-private.jsonl') != audit['games_sha256']):
            raise ValueError('Frozen, audited development required')
        if not previous['eligible_for_confirmation']:
            write(out / 'status.json', {'status': 'skipped', 'reason': 'no_joint_speed_and_point_gain'}); return
    protocol = snapshot(cfg); protocol['checkpoint_sha256'] = digest(checkpoint)
    cached = []
    if cfg.get('reference_cache_out'):
        cached, receipt = load_reference(cfg['reference_cache_out'], cfg, protocol)
        protocol['reference_cache'] = receipt
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text()) != protocol:
        raise ValueError('Resume requires the same candidate and protocol')
    write(out / 'protocol.json', protocol)
    ordinary = load_model(cfg['discard_checkpoint']); calling = load_model(cfg['call_checkpoint'])
    model = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if model.get('base_and_LoRA_frozen') is not True or model['model_files'] != protocol['model_files']:
        raise ValueError('Candidate and reference frozen encoders differ')
    heads = {}
    for scope, role in [('ordinary', 'discard'), ('calling', 'call')]:
        expected = model['model_files'][role]
        for path, h in expected.items():
            if digest(Path(path)) != h: raise ValueError('Candidate feature encoder changed')
        meta = torch.load(Path(cfg[role + '_checkpoint']) / 'head.pt', map_location='cpu', weights_only=True)
        heads[scope] = make_head(meta, 'cuda').eval().requires_grad_(False)
    select_heads(heads, model['heads'], cfg)
    context = model.get('kan_context')
    if context is not None:
        if activation not in ('kan_windows','call_windows') or model['decision_filter'] != activation:
            raise ValueError('Context component requires its registered activation')
        if tuple(model['trained_scopes']) != scopes:
            raise ValueError('Context component differs from its registered decision scope')
        for scope, role in [('ordinary','discard'),('calling','call')]:
            original = torch.load(Path(cfg[role + '_checkpoint'])/'head.pt',map_location='cpu',weights_only=True)['head']
            if any(not torch.equal(v.cpu(),original[k])
                    for k,v in model['heads'][scope].items()):
                raise ValueError('Context component must preserve original pointer heads')
        wrapper = KanContextPolicyClient if activation == 'kan_windows' else CallContextPolicyClient
        router = wrapper(HybridPolicy(ordinary,calling),context)
    else:
        if activation == 'call_windows': raise ValueError('Call-window activation requires a context component')
        router = (KanPolicyClient(ordinary, calling, heads) if activation == 'kan_windows'
            else HeadPolicyClient(ordinary, calling, heads))
    checked = {'ordinary': 0, 'calling': 0}
    if activation == 'kan_windows': checked['ordinary_nonkan'] = 0
    if activation == 'call_windows': checked['ordinary_kan'] = 0
    for path, h in parity_sources(fit).items():
        if digest(Path(path)) != h: raise ValueError('TRAIN parity source changed')
        if Path(path).suffix != '.pt': continue
        for row in torch.load(path, map_location='cpu', weights_only=False)['rows']:
            scope = row['scope']
            key = ('ordinary_nonkan' if activation == 'kan_windows' and scope == 'ordinary'
                and not is_kan_request(row['public_request']) else scope)
            if activation == 'call_windows' and scope == 'ordinary' and is_kan_request(row['public_request']):
                key = 'ordinary_kan'
            if checked[key] >= 2: continue
            head = (ordinary.runtime.model.head if key == 'ordinary_nonkan'
                or (activation == 'call_windows' and scope == 'ordinary') else heads[scope])
            with torch.no_grad():
                expected = head(row['decide'].cuda(), row['options'].cuda()).softmax(-1).cpu()
            if context is not None and key == ('ordinary' if activation == 'kan_windows' else 'calling'):
                expected = torch.tensor(context_probabilities(row['public_request'],expected.tolist(),context))
            p = router.system_one(**row['public_request'])['answers']['action']['probabilities']
            if not torch.allclose(expected, torch.tensor(list(p.values())), atol=2e-5, rtol=2e-4):
                raise ValueError('Updated head and native probabilities disagree')
            checked[key] += 1
        if all(n == 2 for n in checked.values()): break
    if not all(n == 2 for n in checked.values()): raise ValueError('Two-policy native parity missing')
    write(out / 'native-parity.json', checked)
    if cfg.get('preflight_only',False):
        write(out/'status.json',{'status':'native_parity_complete','native_routes':checked,
            'complete_matches_simulated':0,'independent_strength_proven':False})
        return
    agents = {'reference': PolicyAgent(HybridPolicy(ordinary, calling)), 'candidate': PolicyAgent(router)}
    strong = PolicyAgent(ordinary); raw = out / 'games-private.jsonl'
    if not raw.exists() and cached:
        with raw.open('w') as stream:
            for row in cached: stream.write(json.dumps(row) + '\n')
    rows = [json.loads(l) for l in raw.read_text().splitlines()] if raw.exists() else []
    done = {(r['suite'],r['seed'],r['seat'],r['policy']) for r in rows}
    expected = {(s,n,p,k) for s in SUITES for n in range(cfg['seed_start'],cfg['seed_start']+cfg['seeds']) for p in range(4) for k in POLICIES}
    if len(done) != len(rows) or not done <= expected:
        raise ValueError('Foreign or duplicate existing game')
    with raw.open('a') as stream:
        for seed in range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds']):
            for suite in SUITES:
                for seat in range(4):
                    for name in POLICIES:
                        if (suite,seed,seat,name) in done: continue
                        row = dict(policy=name, **play(agents[name], strong, seed, seat, suite, cfg['fast_gamma']))
                        stream.write(json.dumps(row) + '\n'); stream.flush(); rows.append(row)
                        write(out / 'status.json', {'status': 'evaluating', 'games': len(rows),
                            'expected_games': cfg['seeds'] * 24, 'reused_reference_games': len(cached),
                            'new_games_simulated': len(rows) - len(cached)})
    result = summarize(rows, cfg)
    verify(protocol)
    if digest(checkpoint) != protocol['checkpoint_sha256']: raise ValueError('Candidate changed during evaluation')
    raw = out / 'games-private.jsonl'
    if summarize([json.loads(l) for l in raw.read_text().splitlines()], cfg) != result:
        raise ValueError('Saved game reaggregation differs')
    result['improvement_verified'] = (cfg['stage'] == 'independent_confirmation'
        and result['overall']['candidate_minus_reference']['ci95'][0] > 0)
    result['reference_games_reused'] = len(cached)
    result['new_games_simulated'] = len(rows) - len(cached)
    if 'updated_scopes' in cfg: result['updated_scopes'] = list(scopes)
    if 'candidate_activation' in cfg: result['candidate_activation'] = activation
    write(out / 'result.json', result)
    write(out / 'audit.json', {'status': 'passed_complete', 'games': len(rows),
        'games_sha256': digest(raw), 'checkpoint_sha256': digest(checkpoint)})

if __name__ == '__main__': main()
