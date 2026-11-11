"""Short paired East-match evaluation for a frozen Jev Tempo candidate."""
import argparse
import fcntl
import json
from pathlib import Path
import torch
from jevmahjong.core.io import digest, write
from jevmahjong.policy.routing import HybridPolicy, HeadPolicyClient, load_model, make_head
from jevmahjong.training.provenance import snapshot, verify
from .agents import PolicyAgent
from .game import play
from .metrics import summarize, SUITES, POLICIES


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg = json.loads(args.config.read_text()); out = Path(cfg['out'])
    fit_run = Path(cfg['candidate_run']); fit = json.loads((fit_run / 'result.json').read_text())
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
        if (audit['checkpoint_sha256'] != digest(checkpoint)
                or digest(development / 'games-private.jsonl') != audit['games_sha256']):
            raise ValueError('Frozen, audited development required')
        if not previous['eligible_for_confirmation']:
            write(out / 'status.json', {'status': 'skipped', 'reason': 'no_joint_speed_and_point_gain'}); return
    protocol = snapshot(cfg); protocol['checkpoint_sha256'] = digest(checkpoint)
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text()) != protocol:
        raise ValueError('Resume requires the same candidate and protocol')
    write(out / 'protocol.json', protocol)
    ordinary = load_model(cfg['discard_checkpoint']); calling = load_model(cfg['call_checkpoint'])
    model = torch.load(checkpoint, map_location='cpu', weights_only=True)
    heads = {}
    for scope, role in [('ordinary', 'discard'), ('calling', 'call')]:
        expected = model['model_files'][role]
        for path, h in expected.items():
            if digest(Path(path)) != h: raise ValueError('Candidate feature encoder changed')
        meta = torch.load(Path(cfg[role + '_checkpoint']) / 'head.pt', map_location='cpu', weights_only=True)
        heads[scope] = make_head(meta, 'cuda').eval().requires_grad_(False)
        heads[scope].load_state_dict(model['heads'][scope])
    router = HeadPolicyClient(ordinary, calling, heads)
    checked = {'ordinary': 0, 'calling': 0}
    for path, h in fit['data_sha256'].items():
        if digest(Path(path)) != h: raise ValueError('TRAIN parity source changed')
        for row in torch.load(path, map_location='cpu', weights_only=False)['rows']:
            scope = row['scope']
            if checked[scope] >= 2: continue
            with torch.no_grad():
                expected = heads[scope](row['decide'].cuda(), row['options'].cuda()).softmax(-1).cpu()
            p = router.system_one(**row['public_request'])['answers']['action']['probabilities']
            if not torch.allclose(expected, torch.tensor(list(p.values())), atol=2e-5, rtol=2e-4):
                raise ValueError('Updated head and native probabilities disagree')
            checked[scope] += 1
        if all(n == 2 for n in checked.values()): break
    if not all(n == 2 for n in checked.values()): raise ValueError('Two-policy native parity missing')
    write(out / 'native-parity.json', checked)
    agents = {'reference': PolicyAgent(HybridPolicy(ordinary, calling)), 'candidate': PolicyAgent(router)}
    strong = PolicyAgent(ordinary); raw = out / 'games-private.jsonl'
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
                            'expected_games': cfg['seeds'] * 24})
    result = summarize(rows, cfg)
    verify(protocol)
    if digest(checkpoint) != protocol['checkpoint_sha256']: raise ValueError('Candidate changed during evaluation')
    raw = out / 'games-private.jsonl'
    if summarize([json.loads(l) for l in raw.read_text().splitlines()], cfg) != result:
        raise ValueError('Saved game reaggregation differs')
    result['improvement_verified'] = (cfg['stage'] == 'independent_confirmation'
        and result['overall']['candidate_minus_reference']['ci95'][0] > 0)
    write(out / 'result.json', result)
    write(out / 'audit.json', {'status': 'passed_complete', 'games': len(rows),
        'games_sha256': digest(raw), 'checkpoint_sha256': digest(checkpoint)})

if __name__ == '__main__': main()
