"""Explicitly diagnostic self-continuation of unqualified research checkpoints."""
import argparse
import gc
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

REVIEWED_TRAIN_CLI_CHANGE=(
    '849149df25a159e6a1a0e1de4ffea5d636c1e30783d6f4de2bae443b9cc03ca6',
    '31e3250c534ab8ded261c2be8edffd202f09540d6b109d5beb4db15a503d6ab7')


def baseline_dependency_matches(path,before,after):
    if before==after:return True
    # The reviewed CLI pair adds only reuse-fourth-manifest dispatch. This
    # executable is not imported by ModelAgent or rollout. Pin both bytes;
    # further CLI changes still need a separate review.
    if path=='tools/train_endgame.py':return (before,after)==REVIEWED_TRAIN_CLI_CHANGE
    reviewed={f'jevmahjong/endgame/{name}.py' for name in ('context','policy','prepare','reference','mix','fit','futures')}
    return path in reviewed


def validate_configuration(config):
    if (config.get('schema')!='consumed-development-self-continuation-diagnostic-v1'
            or config.get('diagnostic_only') is not True
            or any(config.get(k) is not False for k in ('fit_registered','promotion_registered',
                'independent_confirmation','export_registered','raw_publication'))
            or config.get('public_quota')!={'third':4,'fourth':4,'first':2,'second':2}
            or config.get('per_seed_cap')!=1 or config.get('holding_nonwin_only') is not True
            or config.get('terminal_step_cap')!=30000
            or config.get('input_version')!='endgame-trailing-public-v1'
            or config.get('baseline_cache_reuse') is not True
            or config.get('baseline_control')!='first_hash_selected_nonwin_holding_position'):
        raise ValueError('Only the registered non-promoting consumed-pool diagnostic is allowed')
    specs=config.get('candidates',{})
    if set(specs)!= {'joint_trial10','head_trial11'} or {k:v['step'] for k,v in specs.items()}!={'joint_trial10':72,'head_trial11':48}:
        raise ValueError('The two preregistered research steps changed')


def select_manifest(rows,quota,prefix):
    from jevmahjong.endgame.collect import digest
    remaining=dict(quota);used=set();selected=[]
    def key(row):return digest([prefix,row['cohort'],row['seed'],row['step'],row['seat'],row['request_sha256']])
    def feasible(excluded,needed):
        ranks=list(needed)
        available={rank:{r['seed'] for r in rows if r['stratum']==rank and r['seed'] not in excluded} for rank in ranks}
        return all(len(set().union(*(available[r] for r in subset)))>=sum(needed[r] for r in subset)
                   for size in range(1,len(ranks)+1) for subset in itertools.combinations(ranks,size))
    if not feasible(used,remaining):raise ValueError('Insufficient distinct consumed sources for registered quotas')
    while sum(remaining.values()):
        added=False
        for rank in sorted(remaining):
            if not remaining[rank]:continue
            for row in sorted((r for r in rows if r['stratum']==rank and r['seed'] not in used),key=key):
                needed=dict(remaining);needed[rank]-=1
                if feasible(used|{row['seed']},needed):
                    selected.append(row);used.add(row['seed']);remaining=needed;added=True;break
        if not added:raise ValueError('Public quota assignment failed')
    return selected


def validate_cached_outcome(row):
    from jevmahjong.endgame.context import placements,placement_objective
    b,seat=row['baseline'],row['seat']
    if (not b['terminal_ledger_passed'] or placements(b['final_scores'])[seat]!=b['rank']
            or placement_objective(b['rank'])!=b['utility']
            or b['net_points_from_window']!=b['final_scores'][seat]-b['start_scores'][seat]):
        raise ValueError('Cached baseline score, rank, utility or ledger is inconsistent')


def audit(config):
    validate_configuration(config)
    import torch
    from riichienv import GameRule,RiichiEnv
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.state import build_request
    from jevmahjong.endgame.collect import digest,write
    from jevmahjong.endgame.context import placements,serving_request
    from jevmahjong.endgame.coverage import public_stratum
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.prepare import checksum
    from jevmahjong.endgame.evaluate import rollout,point_and_risk_summary
    from jevmahjong.endgame.policy import ModelAgent
    from jevmahjong.training.futures import replay_prefix
    from tools.evaluate_endgame_pursuit import select_targets
    from tools.evaluate_endgame_reused_development import scope_preflight_evidence
    out,old,source_run=map(Path,(config['out'],config['reference_run'],config['standard_source_run']))
    if (out/'protocol-private.json').exists():raise ValueError('Inspect existing diagnostic before any restart')
    if out.resolve() in {old.resolve(),source_run.resolve(),*(Path(s['run']).resolve() for s in config['candidates'].values())}:
        raise ValueError('Diagnostic output must not modify a source run')
    old_config=json.loads((old/'config-private.json').read_text())
    if (old_config['development_seed_start']!=2026117900 or old_config['development_groups']!=16
            or old_config['terminal_step_cap']!=30000):raise ValueError('Cached development schedule changed')
    baseline_path=Path(old_config['checkpoint']);baseline_id=bundle_identity(baseline_path)
    from desktop.jev_sidecar.unified_policy import identity
    released=identity()
    if baseline_id['identity_sha256'] not in (released['source_identity_sha256'],released['identity_sha256']):
        raise ValueError('The control is not exact released v8')
    general=json.loads((old/'development-result-private.json').read_text())
    targeted=json.loads((old/'targeted-development-result-private.json').read_text())
    target_protocol=json.loads((old/'targeted-development-protocol-private.json').read_text())
    if general['baseline_identity']!=baseline_id or targeted['baseline_identity']!=baseline_id:
        raise ValueError('Cached baseline checkpoint identity changed')
    old_launch=json.loads((old/'fit-launch-protocol-private.json').read_text())
    reviewed_changes={}
    for path,sha in old_launch['code_sha256'].items():
        current=checksum(ROOT/path)
        if not baseline_dependency_matches(path,sha,current):raise ValueError('Baseline runtime changed: '+path)
        if current!=sha:reviewed_changes[path]={'original':sha,'current':current}
    source_hashes={};sources={};entries={};public=[]
    target_root=old/'targeted-development'
    target_sources=[]
    for path in sorted((target_root/'sources').glob('*.json')):
        source=json.loads(path.read_text());source_hashes[str(path)]=checksum(path)
        if source['profile']!='dama':raise ValueError('Target source profile changed')
        for position in source['selected']:
            if public_stratum(position['public_request'])!=position['coverage_stratum']:
                raise ValueError('Target public selection changed')
        target_sources.append(source);sources[('targeted',source['seed'])]=source
    if len(target_sources)!=256 or {s['seed'] for s in target_sources}!=set(range(2026119900,2026120156)):
        raise ValueError('Targeted consumed source set changed')
    target_positions=select_targets(target_sources,32)
    if target_positions!=json.loads((target_root/'selected-positions-private.json').read_text()) or len(target_positions)!=21:
        raise ValueError('Original consumed target manifest changed')
    standard=[]
    for seed in range(2026117900,2026117916):
        path=source_run/'development-sources'/f'{seed}.json'
        source=json.loads(path.read_text());source_hashes[str(path)]=checksum(path)
        if source['seed']!=seed or source['schema']!='endgame-native-source-v1':raise ValueError('Standard cached source differs')
        sources[('standard',seed)]=source;standard.extend(source['selected'])
    if len(standard)!=general['paired_positions']:raise ValueError('Standard window count changed')
    cache_hashes={}
    for cohort,positions in (('standard',standard),('targeted',target_positions)):
        for position in positions:
            seed,seat,step=position['seed'],position['seat'],position['step']
            directory=old/'development' if cohort=='standard' else target_root/'rollouts'
            path=directory/f'{seed}-{step}-{seat}.json';cached=json.loads(path.read_text())
            if (cached['seed'],cached['step'],cached['seat'])!=(seed,step,seat):raise ValueError('Cached window identity changed')
            validate_cached_outcome(cached)
            payload=position['request']
            if digest(payload)!=position['request_sha256']:raise ValueError('Current public request digest changed')
            if cohort=='targeted' and digest(position['public_request'])!=position['public_request_sha256']:
                raise ValueError('Enriched target public request changed')
            rank=placements(payload['state']['scores'])[seat]
            if rank!=cached['initial_rank'] or payload['state']['scores']!=cached['baseline']['start_scores']:
                raise ValueError('Cached initial public scores or placement differ')
            if cohort=='standard' and cached['baseline_identity']!=baseline_id:
                raise ValueError('Standard baseline identity differs')
            if cohort=='targeted' and (cached['protocol_sha256']!=digest(target_protocol)
                    or cached['source_request_sha256']!=position['request_sha256']
                    or cached['manifest_sha256']!=digest(target_positions)):
                raise ValueError('Target cache protocol or manifest changed')
            cache_hashes[str(path)]=checksum(path)
            kinds={v['type'] for v in payload['questions']['action']['criteria'].values()}
            if cohort=='standard' and (rank not in (1,2) or 'hora' in kinds):continue
            if cohort=='targeted' and rank not in (3,4):continue
            stratum={1:'first',2:'second',3:'third',4:'fourth'}[rank]
            row={'cohort':cohort,'seed':seed,'step':step,'seat':seat,'stratum':stratum,
                'request_sha256':position['request_sha256'],'nonwin':'hora' not in kinds}
            key=digest(row);public.append(row);entries[key]=(position,cached)
    selected=select_manifest(public,config['public_quota'],config['selection_hash_prefix'])
    if len(selected)!=12 or len({r['seed'] for r in selected})!=12:raise ValueError('Registered12-source selection changed')
    control=min((r for r in selected if r['stratum'] in ('first','second') and r['nonwin']),
        key=lambda r:digest([config['selection_hash_prefix'],r]))
    protected={};candidate_paths={};candidate_ids={};scope={}
    for name,spec in config['candidates'].items():
        run=Path(spec['run']);fit_path=run/'fit-result-private.json';fit=json.loads(fit_path.read_text())
        run_config=json.loads((run/'config-private.json').read_text())
        if (fit['selection']['step']!=0 or not fit['actual_updates_verified'] or fit['validation'][-1]['step']!=spec['step']
                or run_config['input_version']!=config['input_version'] or Path(run_config['checkpoint']).resolve()!=baseline_path.resolve()):
            raise ValueError('Expected inspected unqualified research checkpoint: '+name)
        expected_scope='head_only' if name=='head_trial11' else 'lora_and_head'
        if run_config.get('trainable_scope','lora_and_head')!=expected_scope:
            raise ValueError('Research parameter scope changed: '+name)
        ready=json.loads((run/'preflight-status-private.json').read_text())
        scope[name]=scope_preflight_evidence(ROOT,run,ready)
        candidate_paths[name]=run/'training-run/checkpoints'/f"step-{spec['step']:06d}"
        candidate_ids[name]=bundle_identity(candidate_paths[name])
        for path in (fit_path,run/'training-run/selection.json',run/'config-private.json'):
            protected[str(path)]=checksum(path)
    sys.path.insert(0,old_config['upstream_source'])
    upstream=Path(old_config['upstream_source'])
    paths=sorted(set(ROOT.glob('jevmahjong/**/*.py'))|set(ROOT.glob('desktop/jev_sidecar/**/*.py'))
        |set((upstream/'jevany').glob('**/*.py'))|{Path(__file__).resolve()})
    protocol={'configuration':config,'selected':selected,'control':control,'baseline_identity':baseline_id,
        'candidate_identities':candidate_ids,'scope_evidence':scope,'source_hashes':source_hashes,
        'cache_hashes':cache_hashes,'protected_fit_files':protected,'code_sha256':{str(p):checksum(p) for p in paths},
        'reviewed_baseline_unaffected_changes':reviewed_changes,
        'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','peft','riichienv','mahjong')},
        'diagnostic_only':True,'promotion_registered':False,'optimizer_updates':0}
    write(out/'selected-positions-private.json',selected);write(out/'protocol-private.json',protocol)
    def window(row):
        source=sources[(row['cohort'],row['seed'])];position,cached=entries[digest(row)]
        env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(row['seed'],'spaced-v1'))
        replay_prefix(env,source['mjai_log'],row['step'])
        if list(env.scores())!=cached['baseline']['start_scores'] or placements(list(env.scores()))[row['seat']]!=cached['initial_rank']:
            raise ValueError('Native prefix score or rank differs')
        obs=env.get_observations()[row['seat']];request,_,_=build_request(obs)
        request['state']['hand_number']=int(obs.kyoku_index)+1
        if digest(serving_request(request))!=position['request_sha256']:
            raise ValueError('Reconstructed current public request differs')
        return env,cached
    from jevmahjong.policy.routing import load_model
    control_env,control_cached=window(control)
    baseline=ModelAgent(load_model(baseline_path),placement_aware=False)
    reproduced=rollout(control_env,control['seat'],control['seed'],baseline,30000)
    write(out/'baseline-control-private.json',{'position':control,'actual':reproduced,
        'expected':control_cached['baseline'],'exactly_equal':reproduced==control_cached['baseline']})
    if reproduced!=control_cached['baseline']:raise ValueError('Fresh baseline replay did not exactly reproduce consumed cache')
    del baseline;gc.collect();torch.cuda.empty_cache()
    results={}
    for name,path in candidate_paths.items():
        learner=ModelAgent(load_model(path),placement_aware=True,input_version=config['input_version']);rows=[]
        for position in selected:
            env,cached=window(position)
            result={k:cached[k] for k in ('seed','seat','step','initial_rank','baseline')}
            result.update(candidate=rollout(env,position['seat'],position['seed'],learner,30000),
                cohort=position['cohort'],stratum=position['stratum'],protocol_sha256=digest(protocol),
                diagnostic_only=True,baseline_reused=True)
            rows.append(result);write(out/name/f"{position['seed']}-{position['step']}-{position['seat']}.json",result)
            write(out/'status-private.json',{'status':'diagnostic_self_continuation','candidate':name,
                'positions_complete':len(rows),'positions_total':12,'promotion_registered':False})
        by_rank={}
        for stratum,cutoff in (('third',2),('fourth',3),('first',2),('second',2)):
            group=[r for r in rows if r['stratum']==stratum]
            by_rank[stratum]={'positions':len(group),**{policy:{'success_count':sum(r[policy]['rank']<=cutoff for r in group),
                'top2_count':sum(r[policy]['rank']<=2 for r in group),
                'mean_utility':sum(r[policy]['utility'] for r in group)/len(group)} for policy in ('baseline','candidate')}}
        results[name]={'positions':12,'by_initial_rank':by_rank,
            'mean_terminal_utility_gain':sum(r['candidate']['utility']-r['baseline']['utility'] for r in rows)/12,
            'points_and_risk':point_and_risk_summary(rows),'diagnostic_only':True}
        del learner;gc.collect();torch.cuda.empty_cache()
    for field in ('source_hashes','cache_hashes','protected_fit_files','code_sha256'):
        for path,sha in protocol[field].items():
            if checksum(path)!=sha:raise ValueError('Read-only diagnostic input changed: '+path)
    if bundle_identity(baseline_path)!=baseline_id or any(bundle_identity(candidate_paths[k])!=v for k,v in candidate_ids.items()):
        raise ValueError('Read-only checkpoint changed')
    summary={'status':'consumed_self_continuation_diagnostic_complete','source_groups':12,'baseline_replays':1,
        'baseline_outcomes_reused':12,'new_candidate_rollouts':24,'results':results,
        'diagnostic_only':True,'eligible_for_promotion':False,'independent_test':False,'deployed':False,
        'optimizer_updates':0,'raw_publication':False}
    write(out/'summary-private.json',summary);write(out/'status-private.json',{'status':summary['status']})
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Self-continuation diagnostic requires an allocated GPU job, never login')
    from tools.prepare_native_context_replay import allocation_metadata
    allocation_metadata(subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True),socket.gethostname())
    print(json.dumps(audit(json.loads(args.config.read_text())),indent=2),flush=True)


if __name__=='__main__':main()
