"""Native serving checks and paired East matches for a complete new adapter."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys

from jevmahjong.core.io import digest,write
from jevmahjong.training.provenance import snapshot,verify
from .unified_contract import checked_candidate,assert_same_candidate,verify_files,bundle,read


def preflight(client,candidate):
    from jevmahjong.policy.unified import serving_request
    from jevmahjong.training.base_lora import validation_profile
    rows=[];routes=dict(closed_discard=0,open_discard=0,call_window=0,ankan=0,daiminkan=0,kakan=0)
    saved={r['id']:dict(zip(r['keys'],r['p'])) for r in candidate['selected_validation_rows']};max_error=0.
    for record in candidate['validation_records']:
        request=serving_request(record)
        # Serving must reproduce the training input, including the instruction
        # and public call summaries, while excluding all labels/targets.
        original=dict(state=record['state'],questions={name:{k:v for k,v in q.items()
            if k in ('type','instructions','criteria')} for name,q in record['questions'].items()})
        if request!=original:raise ValueError('Fresh model training/serving input mismatch')
        prediction=client.system_one(**request)['answers']['action']['probabilities']
        keys=list(record['questions']['action']['criteria'])
        if set(prediction)!=set(keys):raise ValueError('Native returned action menu differs')
        expected=saved[record['_meta']['id']]
        if set(expected)!=set(keys):raise ValueError('Saved selected-model validation menu differs')
        error=max(abs(float(prediction[k])-expected[k]) for k in keys);max_error=max(max_error,error)
        if error>2e-5:raise ValueError('Native probability differs from selected model validation: '+record['_meta']['id']+' max_abs_error='+str(error))
        rows.append(dict(id=record['_meta']['id'],keys=keys,p=[float(prediction[k]) for k in keys]))
        phase=record['_meta']['phase']
        if phase in routes:routes[phase]+=1
        kinds={c['type'] for c in record['questions']['action']['criteria'].values()}
        for kind in ('ankan','daiminkan','kakan'):routes[kind]+=kind in kinds
    if any(n<1 for n in routes.values()):raise ValueError('Ordinary/call/three-kan native coverage missing')
    actual=validation_profile(candidate['validation_records'],rows,candidate['training_config']['validation_phase_weights'])
    expected=candidate['selected_validation']
    for phase,value in actual['phase_target_ce'].items():
        if abs(value-expected['phase_target_ce'][phase])>1e-5:raise ValueError('Selected checkpoint native CE differs: '+phase)
    if abs(actual['weighted_target_ce']-expected['weighted_target_ce'])>1e-5:
        raise ValueError('Selected checkpoint native weighted CE differs')
    return dict(status='passed_selected_unified_native_validation',records=len(rows),routes=routes,
        candidate_identity_sha256=candidate['model']['identity_sha256'],selected_step=candidate['selected_step'],
        max_abs_probability_difference=max_error,actual_validation=actual,complete_matches_simulated=0),rows


def run(cfg,resume=False):
    from .metrics import summarize,SUITES,POLICIES
    candidate=checked_candidate(cfg['candidate_run']);train_cfg=candidate['training_config']
    if not isinstance(cfg.get('preflight_only',False),bool):raise ValueError('Boolean native-preflight flag required')
    if cfg.get('candidate_model_id')!=candidate['model_id']:raise ValueError('Prescribed unified model id differs')
    if cfg.get('confirmation_objective')!='hand_win' or cfg.get('reference_cache_out'):
        raise ValueError('Fresh-reference hand-win comparison required')
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=resume)
    lock=(out/'evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reference=cfg['reference'];proto=snapshot({key:reference[key] for key in ('discard_checkpoint','call_checkpoint')})
    proto['config']=cfg;proto['candidate_bundle']=candidate['model'];proto['fit_receipt']=candidate['receipt']
    proto['reference_policy']=reference;proto['source_files']={reference['call_context_baseline']['checkpoint']:reference['call_context_baseline']['sha256']}
    verify_files(proto['source_files'])
    if not cfg.get('preflight_only',False):
        proof=Path(cfg['required_native_preflight']);before=read(proof)
        if (before['status']!='passed_selected_unified_native_validation'
                or before['candidate_identity_sha256']!=candidate['model']['identity_sha256']):
            raise ValueError('Prospective zero-match native preflight required')
        proto['native_preflight_receipt']={str(proof):digest(proof)}
    if cfg['stage']=='independent_confirmation':
        previous=Path(cfg['development_out']);p=read(previous/'protocol.json');r=read(previous/'result.json');a=read(previous/'independent-audit.json')
        if (not r['eligible_for_confirmation'] or a['status']!='passed_complete'
                or p['candidate_bundle']!=candidate['model']
                or any(p['config'].get(k)!=cfg.get(k) for k in ('reference','confirmation_objective','fast_gamma','max_dealin_rate_increase'))
                or a['games_sha256']!=digest(previous/'games-private.jsonl')):
            raise ValueError('Same candidate and independently audited positive development required')
        proto['development_receipt']={str(previous/n):digest(previous/n) for n in ('result.json','protocol.json','independent-audit.json')}
    elif cfg['stage']!='development':raise ValueError('Unknown unified evaluation stage')
    if (out/'protocol.json').exists() and read(out/'protocol.json')!=proto:raise ValueError('Resume candidate or recipe differs')
    write(out/'protocol.json',proto)
    if not candidate['eligible_for_development']:
        write(out/'status.json',dict(status='skipped_no_validation_learning',complete_matches_simulated=0));return
    upstream=Path(train_cfg['upstream_source']);sys.path.insert(0,str(upstream))
    import jevany
    if not Path(jevany.__file__).resolve().is_relative_to(upstream):raise ValueError('Unpinned inference package')
    # All three model loads use the exact pinned pretrained bytes verified by
    # the training receipt; the incumbent loaders retain their original flags.
    os.environ['JEV_BASE_MODEL_DIR']=train_cfg['base_load_path']
    from jevmahjong.policy.routing import load_model,HybridPolicy
    from jevmahjong.policy.unified import UnifiedPolicyClient
    client=UnifiedPolicyClient(load_model(candidate['model']['directory']))
    native,rows=preflight(client,candidate);write(out/'native-preflight.json',native);write(out/'native-rows-private.json',rows)
    if cfg.get('preflight_only',False):
        write(out/'status.json',native);return
    from jevmahjong.policy.call_baseline import load as load_baseline,wrap as wrap_baseline
    from .agents import PolicyAgent
    from .game import play
    ordinary=load_model(reference['discard_checkpoint']);calling=load_model(reference['call_checkpoint'])
    baseline=load_baseline(reference,proto['model_files'])
    agents=dict(reference=PolicyAgent(wrap_baseline(HybridPolicy(ordinary,calling),baseline)),candidate=PolicyAgent(client))
    strong=PolicyAgent(ordinary);raw=out/'games-private.jsonl'
    games=[json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
    done={(r['suite'],r['seed'],r['seat'],r['policy']) for r in games}
    grid={(s,n,p,k) for s in SUITES for n in range(cfg['seed_start'],cfg['seed_start']+cfg['seeds']) for p in range(4) for k in POLICIES}
    if len(done)!=len(games) or not done<=grid:raise ValueError('Foreign or duplicate existing game')
    with raw.open('a') as stream:
        for seed in range(cfg['seed_start'],cfg['seed_start']+cfg['seeds']):
            for suite in SUITES:
                for seat in range(4):
                    for policy in POLICIES:
                        key=(suite,seed,seat,policy)
                        if key in done:continue
                        row=dict(policy=policy,**play(agents[policy],strong,seed,seat,suite,cfg['fast_gamma']))
                        stream.write(json.dumps(row)+'\n');stream.flush();games.append(row)
                        write(out/'status.json',dict(status='evaluating_unified_policy',games=len(games),expected_games=len(grid),candidate_identity_sha256=candidate['model']['identity_sha256']))
    result=summarize(games,cfg);verify(proto);verify_files(proto['source_files']);verify_files(proto['fit_receipt'])
    verify_files(proto['native_preflight_receipt']);verify_files(proto.get('development_receipt',{}))
    if bundle(candidate['model']['directory'])!=candidate['model']:raise ValueError('Adapter/head changed during matches')
    component='unified_continued_adapter_and_pointer_head' if candidate.get('initialization_mode')=='continued' else 'unified_new_adapter_and_pointer_head'
    result.update(candidate_identity_sha256=candidate['model']['identity_sha256'],candidate_component=component,
        reference_games_reused=0,new_games_simulated=len(games),independent_audit_required=True)
    write(out/'evaluation-result.json',result);write(out/'audit.json',dict(status='evaluation_complete_independent_audit_pending',games=len(games),games_sha256=digest(raw),candidate_identity_sha256=candidate['model']['identity_sha256'],
        native_files={str(out/name):digest(out/name) for name in ('native-preflight.json','native-rows-private.json')}))
    write(out/'status.json',dict(status='matches_complete_independent_audit_pending',games=len(games)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    run(read(a.config),a.resume)


if __name__=='__main__':main()
