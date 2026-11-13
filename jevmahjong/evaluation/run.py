"""Short paired East-match evaluation for a frozen Jev Tempo candidate."""
import argparse
import fcntl
import json
from pathlib import Path
import torch
from jevmahjong.core.io import digest, write
from jevmahjong.policy.routing import HybridPolicy, HeadPolicyClient, load_model, make_head
from jevmahjong.policy.kan import KanPolicyClient, is_kan_request
from jevmahjong.policy.kan_context import KanContextPolicyClient, CallContextPolicyClient, DefenseContextPolicyClient, probabilities as context_probabilities
from jevmahjong.policy.defense import DefensePolicyAgent,is_defense_request
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
    if mode not in ('all', 'kan_windows', 'call_windows', 'defense_windows'):
        raise ValueError('Unknown candidate activation')
    if mode == 'kan_windows' and candidate_scopes(config) != ('ordinary',):
        raise ValueError('Kan activation requires only the ordinary updated head')
    if mode == 'call_windows' and candidate_scopes(config) != ('calling',):
        raise ValueError('Call activation requires only the calling scope')
    if mode == 'defense_windows' and (candidate_scopes(config) != ('ordinary',)
            or config.get('candidate_action_admission')!='open_tenpai_genbutsu_v1'):
        raise ValueError('Defense activation requires ordinary scope and explicit expanded admission')
    if trained_filter in ('kan_windows','call_windows','defense_windows') and mode != trained_filter:
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


def parity_anchor_paths(directory, model_files):
    """Parity inputs may come from a fit or a completed native collection."""
    root=Path(directory).resolve();proto_file=root/'protocol.json'
    protocol=json.loads(proto_file.read_text());verify(protocol)
    if protocol['model_files']!=model_files:raise ValueError('Parity encoders differ')
    paths={str(proto_file):digest(proto_file)}
    if (root/'result.json').exists():
        fit_file=root/'result.json';fit=json.loads(fit_file.read_text())
        if fit['status']!='trained':raise ValueError('Completed parity fit required')
        paths[str(fit_file)]=digest(fit_file);paths.update(parity_sources(fit));return paths
    marker_file=root/'collection-result.json';marker=json.loads(marker_file.read_text())
    cfg=protocol['config'];start=cfg['training_seed_start'];count=cfg['training_seeds_per_stage']
    if marker['status']!='collection_complete' or marker['walls']!=count:
        raise ValueError('Complete native parity source required')
    shards=list((root/'walls-private').glob('*.pt'))
    if len(shards)!=count or {int(p.stem) for p in shards}!=set(range(start,start+count)):
        raise ValueError('Native parity wall grid differs')
    rows=0;paths[str(marker_file)]=digest(marker_file)
    for shard in sorted(shards):
        receipt=shard.with_suffix('.audit.json');record=json.loads(receipt.read_text());checksum=digest(shard)
        if record['sha256']!=checksum:raise ValueError('Native parity shard changed')
        rows+=record['rows'];paths[str(shard)]=checksum;paths[str(receipt)]=digest(receipt)
    if rows!=marker['rows']:raise ValueError('Native parity row count differs')
    return paths


def parity_route(row,activation):
    scope=row['scope'];request=row['public_request']
    if activation=='defense_windows' and scope=='ordinary':
        return 'ordinary' if is_defense_request(request) else 'ordinary_kan' if is_kan_request(request) else 'ordinary_other'
    if activation=='kan_windows' and scope=='ordinary' and not is_kan_request(request):return 'ordinary_nonkan'
    if activation=='call_windows' and scope=='ordinary' and is_kan_request(request):return 'ordinary_kan'
    return scope


def parity_rows(path,checksum):
    """Receipts also bind model weights; only native shards contain inputs."""
    path=Path(path)
    if digest(path)!=checksum:raise ValueError('TRAIN parity source changed')
    if path.suffix!='.pt' or path.parent.name!='walls-private':return []
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if not isinstance(payload,dict) or not isinstance(payload.get('rows'),list):
        raise ValueError('Native parity shard rows missing')
    return payload['rows']


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
        if 'hand_win' in (cfg.get('confirmation_objective'),development_protocol['config'].get('confirmation_objective')):
            if any(cfg.get(k)!=development_protocol['config'].get(k) for k in
                    ('confirmation_objective','fast_gamma','max_dealin_rate_increase')):
                raise ValueError('Confirmation must preserve the registered win objective and risk constraint')
        if candidate_scopes(development_protocol['config']) != scopes:
            raise ValueError('Confirmation must preserve development head scopes')
        if candidate_activation(development_protocol['config']) != activation:
            raise ValueError('Confirmation must preserve development activation')
        if development_protocol['config'].get('candidate_action_admission')!=cfg.get('candidate_action_admission'):
            raise ValueError('Confirmation must preserve candidate action admission')
        if (audit['checkpoint_sha256'] != digest(checkpoint)
                or digest(development / 'games-private.jsonl') != audit['games_sha256']):
            raise ValueError('Frozen, audited development required')
        if not previous['eligible_for_confirmation']:
            write(out / 'status.json', {'status': 'skipped', 'reason': 'no_joint_speed_and_point_gain'}); return
    protocol = snapshot(cfg); protocol['checkpoint_sha256'] = digest(checkpoint)
    fit_protocol=json.loads((fit_run/'protocol.json').read_text())
    if protocol.get('call_context_baseline')!=fit_protocol.get('call_context_baseline'):
        raise ValueError('Evaluation must preserve the fit calling continuation baseline')
    if cfg['stage']=='independent_confirmation' and (
            protocol.get('call_context_baseline')!=development_protocol.get('call_context_baseline')):
        raise ValueError('Confirmation must preserve the development calling baseline')
    if cfg.get('required_call_head_audit_config'):
        from .call_head_audit import checked_receipt
        checked,receipt=checked_receipt(cfg['required_call_head_audit_config'],fit_run,fit,protocol['model_files'])
        protocol['fit_audit_receipt']=receipt
        if not cfg.get('preflight_only',False) and not checked['eligible_for_development']:
            write(out/'protocol.json',protocol)
            write(out/'status.json',{'status':'skipped','reason':('no_audited_held_hand_win_and_route_gain'
                if checked.get('selection_objective')=='hand_win' else 'no_audited_held_choice_point_gain'),
                'complete_matches_simulated':0,'independent_strength_proven':False});return
    if activation=='defense_windows' or cfg.get('required_fit_audit_config'):
        from .context_audit import checked_receipt
        checked,receipt=checked_receipt(cfg['required_fit_audit_config'],fit_run,fit,protocol['model_files'])
        protocol['fit_audit_receipt']=receipt
        if not cfg.get('preflight_only',False) and not checked['eligible_for_development']:
            write(out/'protocol.json',protocol)
            write(out/'status.json',{'status':'skipped','reason':('no_audited_held_hand_win_gain'
                if checked.get('selection_objective')=='hand_win' else 'no_audited_held_policy_and_sampled_net_point_gain'),
                'complete_matches_simulated':0,'independent_strength_proven':False});return
    cached = []
    if cfg.get('reference_cache_out'):
        cached, receipt = load_reference(cfg['reference_cache_out'], cfg, protocol)
        protocol['reference_cache'] = receipt
    parity_paths=parity_sources(fit)
    if cfg.get('parity_source_run'):
        external=parity_anchor_paths(cfg['parity_source_run'],protocol['model_files'])
        for path,h in external.items():
            if path in parity_paths and parity_paths[path]!=h:raise ValueError('Conflicting external parity identity')
            parity_paths[path]=h
        protocol['parity_source_receipt']=external
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text()) != protocol:
        raise ValueError('Resume requires the same candidate and protocol')
    write(out / 'protocol.json', protocol)
    ordinary = load_model(cfg['discard_checkpoint']); calling = load_model(cfg['call_checkpoint'])
    model = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if model.get('base_and_LoRA_frozen') is not True or model['model_files'] != protocol['model_files']:
        raise ValueError('Candidate and reference frozen encoders differ')
    if model.get('call_context_baseline')!=protocol.get('call_context_baseline'):
        raise ValueError('Candidate calling baseline differs')
    from jevmahjong.policy.call_baseline import load as load_baseline,wrap as wrap_baseline,transform as baseline_transform,replaces_calling_baseline
    calling_baseline=load_baseline(cfg,protocol['model_files'])
    replaces_baseline=replaces_calling_baseline(model)
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
        if activation not in ('kan_windows','call_windows','defense_windows') or model['decision_filter'] != activation:
            raise ValueError('Context component requires its registered activation')
        if tuple(model['trained_scopes']) != scopes:
            raise ValueError('Context component differs from its registered decision scope')
        for scope, role in [('ordinary','discard'),('calling','call')]:
            original = torch.load(Path(cfg[role + '_checkpoint'])/'head.pt',map_location='cpu',weights_only=True)['head']
            if any(not torch.equal(v.cpu(),original[k])
                    for k,v in model['heads'][scope].items()):
                raise ValueError('Context component must preserve original pointer heads')
        wrapper = {'kan_windows':KanContextPolicyClient,'call_windows':CallContextPolicyClient,
            'defense_windows':DefenseContextPolicyClient}[activation]
        router = wrapper(HybridPolicy(ordinary,calling),context)
    else:
        if activation in ('call_windows','defense_windows'): raise ValueError('Scoped activation requires a context component')
        router = (KanPolicyClient(ordinary, calling, heads) if activation == 'kan_windows'
            else HeadPolicyClient(ordinary, calling, heads))
    if not replaces_baseline:router=wrap_baseline(router,calling_baseline)
    checked = {'ordinary': 0, 'calling': 0}
    if activation == 'kan_windows': checked['ordinary_nonkan'] = 0
    if activation == 'call_windows': checked['ordinary_kan'] = 0
    if activation == 'defense_windows':checked.update(ordinary_other=0,ordinary_kan=0)
    for path, h in parity_paths.items():
        rows=parity_rows(path,h)
        if not rows:continue
        shard_config=None
        if activation=='defense_windows':
            shard_protocol=Path(path).parent.parent/'protocol.json'
            shard_config=json.loads(shard_protocol.read_text())['config']
        for row in rows:
            if shard_config is not None and (row['episode_key'][0]-shard_config['training_seed_start'])%4==3:continue
            scope = row['scope']
            key = parity_route(row,activation)
            if checked[key] >= 2: continue
            head = (ordinary.runtime.model.head if key == 'ordinary_nonkan'
                or (activation == 'call_windows' and scope == 'ordinary') else heads[scope])
            with torch.no_grad():
                expected = head(row['decide'].cuda(), row['options'].cuda()).softmax(-1).cpu()
            if context is not None and key == ('calling' if activation == 'call_windows' else 'ordinary'):
                expected = torch.tensor(context_probabilities(row['public_request'],expected.tolist(),context))
            if not replaces_baseline:expected=baseline_transform(row['public_request'],expected,calling_baseline)
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
    candidate_agent=DefensePolicyAgent(router) if activation=='defense_windows' else PolicyAgent(router)
    agents = {'reference': PolicyAgent(wrap_baseline(HybridPolicy(ordinary, calling),calling_baseline)), 'candidate': candidate_agent}
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
                        if activation=='defense_windows' and name=='candidate':candidate_agent.admissions.clear()
                        row = dict(policy=name, **play(agents[name], strong, seed, seat, suite, cfg['fast_gamma']))
                        if activation=='defense_windows' and name=='candidate':row['defense_admissions']=dict(candidate_agent.admissions)
                        stream.write(json.dumps(row) + '\n'); stream.flush(); rows.append(row)
                        write(out / 'status.json', {'status': 'evaluating', 'games': len(rows),
                            'expected_games': cfg['seeds'] * 24, 'reused_reference_games': len(cached),
                            'new_games_simulated': len(rows) - len(cached)})
    result = summarize(rows, cfg)
    verify(protocol)
    for path,h in protocol.get('parity_source_receipt',{}).items():
        if digest(Path(path))!=h:raise ValueError('External parity source changed during evaluation')
    for path,h in protocol.get('fit_audit_receipt',{}).items():
        if digest(Path(path))!=h:raise ValueError('Fit audit receipt changed during evaluation')
    if digest(checkpoint) != protocol['checkpoint_sha256']: raise ValueError('Candidate changed during evaluation')
    raw = out / 'games-private.jsonl'
    if summarize([json.loads(l) for l in raw.read_text().splitlines()], cfg) != result:
        raise ValueError('Saved game reaggregation differs')
    result['improvement_verified'] = (cfg['stage'] == 'independent_confirmation'
        and (result['hand_win_paired']['ci95'][0]>0 and result['dealin_rate_paired']['gain']<=cfg['max_dealin_rate_increase']
            if cfg.get('confirmation_objective')=='hand_win' else result['overall']['candidate_minus_reference']['ci95'][0]>0))
    result['reference_games_reused'] = len(cached)
    result['new_games_simulated'] = len(rows) - len(cached)
    if 'updated_scopes' in cfg: result['updated_scopes'] = list(scopes)
    if 'candidate_activation' in cfg: result['candidate_activation'] = activation
    write(out / 'result.json', result)
    write(out / 'audit.json', {'status': 'passed_complete', 'games': len(rows),
        'games_sha256': digest(raw), 'checkpoint_sha256': digest(checkpoint)})

if __name__ == '__main__': main()
