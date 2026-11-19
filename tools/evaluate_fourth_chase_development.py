"""Frozen-v8 fourth-only endgame-path development on consumed cached source pools."""
import argparse
from collections import defaultdict
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def validate_config(config):
    expected={'schema':'fourth-scope-consumed-development-v1','out':'runs/endgame-fourth-chase-consumed-development-v1',
        'standard_seed_start':2026117900,'standard_groups':16,'standard_positions':62,
        'targeted_seed_start':2026119900,'targeted_source_groups':256,'targeted_positions':21,
        'baseline_rollouts_reused':83,'baseline_control_rollouts':2,'new_candidate_rollouts':83,
        'checkpoint':'formal_v8_step2088','candidate_fourth_policy':'endgame-fourth-chase-v1',
        'candidate_nonfourth_policy':'released_v8','terminal_step_cap':30000,
        'primary':'fourth_final_top2_source_group_gain','fourth_escape_constraint':'report_only',
        'third_no_drop_each_cohort':True,'holding_max_drop':.05,'evaluation_registered':True}
    if any(config.get(k)!=v for k,v in expected.items()) or any(config.get(k) is not False for k in ('independent_confirmation','raw_publication')):
        raise ValueError('Only the preregistered frozen-v8 fourth-only consumed-pool development is enabled')


def candidate_agent(client):
    from jevmahjong.endgame.fourth_policy import FourthChaseAgent
    agent=FourthChaseAgent(client)
    # Isolate the fourth-place scope. A return to third/leading uses the
    # original released input and guards, rather than the third-place trial.
    agent.base.placement_aware=False
    return agent


def static_path_ablation(records,predictions,reference):
    from jevmahjong.endgame.public_policy import action_keys
    from jevmahjong.endgame.fourth_chase import fourth_chase_action_keys
    arms=defaultdict(list);input_changes=ordinary_scope_changes=0
    for row in records:
        meta=row['_meta'];rid=meta['id']
        if meta.get('source_initial_placement')!=4 or row['questions']['action']['src']!='endgame_terminal':continue
        old_scope=action_keys(row);new_scope=fourth_chase_action_keys(row)
        original=max(old_scope,key=reference[rid]['probabilities'].get)
        rich=max(old_scope,key=predictions[rid].get);new=max(new_scope,key=predictions[rid].get)
        input_changes+=original!=rich;ordinary_scope_changes+=rich!=new
        for name,key in (('released_reference',original),('existing_endgame_input_and_guards',rich),('fourth_legal_ordinary_path',new)):
            arms[name].append({'top2':meta['fourth_chase_top2_probabilities'][key],
                              'fourth':1-meta['terminal_escape4_probabilities'][key]})
    if len(arms['released_reference'])!=51:raise ValueError('Static path comparison must keep all51 fourth validation positions')
    return {'positions':51,'arms':{name:{k:sum(v[k] for v in values)/len(values) for k in ('top2','fourth')}
        for name,values in arms.items()},'input_and_win_scope_choice_changes':input_changes,
        'ordinary_scope_only_choice_changes':ordinary_scope_changes,'finite_old_validation_not_strength':True}


def summarize(records):
    from jevmahjong.endgame.evaluate import interval,point_and_risk_summary
    fourth=[r for r in records if r['initial_rank']==4];third=[r for r in records if r['initial_rank']==3]
    if not fourth or not third:raise ValueError('Development must retain fourth and third coverage')
    def rates(rows):
        return {policy:{'top2_count':sum(r[policy]['rank']<=2 for r in rows),
            'top2_rate':sum(r[policy]['rank']<=2 for r in rows)/len(rows),
            'fourth_count':sum(r[policy]['rank']==4 for r in rows),
            'fourth_rate':sum(r[policy]['rank']==4 for r in rows)/len(rows)} for policy in ('baseline','candidate')}
    grouped=defaultdict(list)
    for r in fourth:grouped[r['seed']].append(int(r['candidate']['rank']<=2)-int(r['baseline']['rank']<=2))
    differences=[sum(v)/len(v) for v in grouped.values()]
    cohorts={}
    for name in ('standard','targeted'):
        rows=[r for r in records if r['cohort']==name];rank3=[r for r in rows if r['initial_rank']==3]
        rank4=[r for r in rows if r['initial_rank']==4];leading=[r for r in rows if r['initial_rank']<=2]
        if not rank3 or not rank4:raise ValueError('Keep both trailing ranks in each consumed pool')
        cohorts[name]={'positions':len(rows),'third':dict(positions=len(rank3),**rates(rank3)),
            'fourth':dict(positions=len(rank4),**rates(rank4))}
        if leading:cohorts[name]['holding']=dict(positions=len(leading),**rates(leading))
    fourth_rates=rates(fourth);window_gain=fourth_rates['candidate']['top2_rate']-fourth_rates['baseline']['top2_rate']
    group_gain=sum(differences)/len(differences)
    third_safe=all(c['third']['candidate']['top2_rate']>=c['third']['baseline']['top2_rate'] for c in cohorts.values())
    holding=cohorts['standard']['holding'];holding_safe=holding['candidate']['top2_rate']>=holding['baseline']['top2_rate']-.05
    return {'positions':len(records),'fourth_positions':len(fourth),'fourth_source_groups':len(grouped),
        'fourth_rates':fourth_rates,'fourth_window_top2_gain':window_gain,'fourth_source_group_top2_gain':group_gain,
        'source_group_top2_gain_ci95':interval(differences),'cohorts':cohorts,
        'third_no_drop_each_cohort':third_safe,'holding_guard_passed':holding_safe,
        'eligible_for_new_confirmation':group_gain>0 and window_gain>0 and third_safe and holding_safe,
        'fourth_escape_is_reporting_only':True,'points_and_dealins':point_and_risk_summary(records),
        'not_independent_confirmation':True}


def evaluate(config):
    validate_config(config)
    from jevmahjong.endgame.collect import digest,write
    from jevmahjong.endgame.prepare import checksum
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.context import placements,placement_objective,serving_request
    from jevmahjong.endgame.coverage import public_stratum
    from tools.evaluate_endgame_pursuit import select_targets
    out=Path(config['out']);old=Path(config['baseline_reference_run']);fit_root=Path(config['source_fit_run'])
    if any((out/name).exists() for name in ('protocol-private.json','summary-private.json')):
        raise ValueError('Inspect an existing evaluation instead of silently restarting')
    fit_audit=json.loads((fit_root/'postfit-summary-private.json').read_text())
    if (fit_audit['last_step']!=2 or fit_audit['selected_step']!=0
            or not fit_audit['normal_service_control']['passed']):raise ValueError('Expected the completed reviewed16 fit')
    ready=json.loads((fit_root/'ready-private.json').read_text());source_config=ready['source_config']
    parent=Path(source_config['checkpoint']);identity=bundle_identity(parent)
    if identity!=ready['parent_identity']:raise ValueError('Formal source weights changed')
    old_general=json.loads((old/'development-result-private.json').read_text())
    old_target=json.loads((old/'targeted-development-result-private.json').read_text())
    target_protocol=json.loads((old/'targeted-development-protocol-private.json').read_text())
    if old_general['baseline_identity']!=identity or old_target['baseline_identity']!=identity:
        raise ValueError('Cached baseline weights differ')
    original_launch=json.loads((old/'fit-launch-protocol-private.json').read_text())
    from jevmahjong.endgame.prepare import read_jsonl
    static_records=read_jsonl(fit_root/'validation.jsonl')
    if checksum(fit_root/'validation.jsonl')!=ready['data_sha256'][str(fit_root/'validation.jsonl')]:
        raise ValueError('Prepared static validation changed')
    static_predictions_path=fit_root/'training-run/predictions/step-000000-private.json'
    static_reference_path=fit_root/'reference-targets-private.json'
    ablation=static_path_ablation(static_records,json.loads(static_predictions_path.read_text()),
        json.loads(static_reference_path.read_text())['rows'])
    if any(abs(ablation['arms']['fourth_legal_ordinary_path'][k]-ready['initial_validation']['fourth'][k])>1e-12
           for k in ('top2','fourth')):raise ValueError('Static path decomposition differs from audited baseline0')
    critical_desktop={'unified_policy.py','call_policy.py','call_features.py','call_feature_helpers.py','tiles.py'}
    baseline_checked={}
    for name,expected in original_launch['code_sha256'].items():
        path=Path(name);relative=str(path.relative_to(ROOT)) if path.is_absolute() else name
        critical=(relative.startswith('jevmahjong/') and not relative.startswith('jevmahjong/endgame/')
            or relative in ('jevmahjong/endgame/collect.py','jevmahjong/endgame/ledger.py','jevmahjong/endgame/evaluate.py')
            or relative.startswith('desktop/jev_sidecar/') and Path(relative).name in critical_desktop)
        if critical:
            if checksum(ROOT/relative)!=expected:raise ValueError('Baseline/source runtime changed: '+relative)
            baseline_checked[relative]=expected
    if len(baseline_checked)<20:raise ValueError('Baseline identity checks unexpectedly incomplete')
    target_dir=old/'targeted-development';target_sources=[json.loads(p.read_text()) for p in sorted((target_dir/'sources').glob('*.json'))]
    if len(target_sources)!=256 or {s['seed'] for s in target_sources}!=set(range(2026119900,2026120156)):
        raise ValueError('Consumed targeted source pool changed')
    for source in target_sources:
        if source['profile']!='dama':raise ValueError('Target source profile changed')
        if any(public_stratum(p['public_request'])!=p['coverage_stratum'] for p in source['selected']):
            raise ValueError('Target source public selection changed')
    selected=select_targets(target_sources,32)
    if len(selected)!=21 or selected!=json.loads((target_dir/'selected-positions-private.json').read_text()):
        raise ValueError('Keep every original21 targeted window')
    lookup={s['seed']:s for s in target_sources};entries=[];paths=set()
    for index in range(16):
        path=Path(config['standard_source_root'])/f'{2026117900+index}.json';paths.add(path);source=json.loads(path.read_text())
        if source['seed']!=2026117900+index:raise ValueError('Standard source schedule changed')
        for position in source['selected']:entries.append(('standard',source,position,old/'development'))
    if len(entries)!=62 or old_general['paired_positions']!=62:raise ValueError('Keep every original62 standard window')
    for position in selected:entries.append(('targeted',lookup[position['seed']],position,target_dir/'rollouts'))
    paths.update(target_dir/'sources'/f"{s['seed']}.json" for s in target_sources)
    paths.add(target_dir/'selected-positions-private.json')
    fitted=json.loads((Path(source_config['out'])/'selected-positions-private.json').read_text())
    fitted_seeds={r['seed'] for r in fitted}
    native=json.loads((Path(source_config['native_context_root'])/'selected-positions-private.json').read_text())
    fitted_seeds.update(r['seed'] for r in native)
    if fitted_seeds&{s['seed'] for _,s,_,_ in entries}:raise ValueError('Consumed development overlaps fitted source groups')
    cached={}
    for cohort,source,position,directory in entries:
        name=f"{position['seed']}-{position['step']}-{position['seat']}";path=directory/(name+'.json');paths.add(path)
        row=json.loads(path.read_text());seat=position['seat'];baseline=row['baseline']
        if (row['seed'],row['step'],row['seat'])!=(position['seed'],position['step'],seat):raise ValueError('Cached window changed')
        if (not baseline['terminal_ledger_passed'] or placements(baseline['final_scores'])[seat]!=baseline['rank']
                or placement_objective(baseline['rank'])!=baseline['utility']
                or baseline['net_points_from_window']!=baseline['final_scores'][seat]-baseline['start_scores'][seat]):
            raise ValueError('Cached terminal baseline failed accounting')
        if cohort=='standard' and row['baseline_identity']!=identity:raise ValueError('Cached standard source model changed')
        if cohort=='targeted' and (row['protocol_sha256']!=digest(target_protocol)
                or row['source_request_sha256']!=position['request_sha256'] or row['manifest_sha256']!=digest(selected)):
            raise ValueError('Cached target protocol/source changed')
        cached[(cohort,name)]=row
    sys.path.insert(0,source_config['upstream_source'])
    paths.update(ROOT.glob('jevmahjong/**/*.py'));paths.update(ROOT.glob('desktop/jev_sidecar/**/*.py'))
    paths.update(ROOT/name for name in ('tools/evaluate_fourth_chase_development.py','tools/evaluate_endgame_pursuit.py',
        'configs/training/endgame-fourth-scope-consumed-development-v1.json','tests/core/test_fourth_scope_development.py'))
    paths.update(old/name for name in ('fit-launch-protocol-private.json','development-result-private.json',
        'targeted-development-result-private.json','targeted-development-protocol-private.json'))
    paths.update(fit_root/name for name in ('ready-private.json','postfit-summary-private.json'))
    paths.update((fit_root/'validation.jsonl',static_predictions_path,static_reference_path))
    controls=[]
    for cohort in ('standard','targeted'):
        values=[e for e in entries if e[0]==cohort and cached[(cohort,f"{e[2]['seed']}-{e[2]['step']}-{e[2]['seat']}")]['initial_rank']==4]
        controls.append(min(values,key=lambda e:digest([e[0],e[2]['seed'],e[2]['step'],e[2]['seat'],'baseline-control-v1'])))
    protocol={'configuration':config,'candidate_weight_identity':identity,'baseline_weight_identity':identity,
        'candidate_policy':'frozen_v8_fourth_scope_only_nonfourth_released','baseline_runtime_hashes_verified':baseline_checked,
        'source_sha256':{str(p.resolve()):checksum(p) for p in sorted(paths)},
        'baseline_control_ids':[[c,p['seed'],p['step'],p['seat']] for c,s,p,d in controls],
        'versions':{n:importlib.metadata.version(n) for n in ('torch','transformers','peft','riichienv','mahjong')},
        'baseline_rollouts_reused':83,'candidate_rollouts':83,'independent_confirmation':False}
    write(out/'protocol-private.json',protocol)
    write(out/'static-path-ablation-private.json',ablation)
    from riichienv import GameRule,RiichiEnv
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.state import build_request
    from jevmahjong.training.futures import replay_prefix
    from jevmahjong.endgame.evaluate import rollout
    from jevmahjong.endgame.policy import ModelAgent
    from jevmahjong.policy.routing import load_model
    client=load_model(parent)
    if client.runtime.inference_options.prefix_cache_size!=0:raise ValueError('Inference prefix cache must remain disabled')
    def environment(entry):
        cohort,source,position,_=entry;seat=position['seat'];name=f"{position['seed']}-{position['step']}-{seat}"
        env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(position['seed'],'spaced-v1'))
        replay_prefix(env,source['mjai_log'],position['step']);stored=cached[(cohort,name)]
        if list(env.scores())!=stored['baseline']['start_scores'] or placements(list(env.scores()))[seat]!=stored['initial_rank']:
            raise ValueError('Replayed source scores/rank differ')
        obs=env.get_observations()[seat];request,_,_=build_request(obs);request['state']['hand_number']=int(obs.kyoku_index)+1
        if digest(serving_request(request))!=position['request_sha256']:raise ValueError('Replayed public request changed')
        return env
    checked=[]
    for entry in controls:
        cohort,_,position,_=entry;name=f"{position['seed']}-{position['step']}-{position['seat']}"
        actual=rollout(environment(entry),position['seat'],position['seed'],ModelAgent(client,placement_aware=False),30000)
        if actual!=cached[(cohort,name)]['baseline']:raise ValueError('Actual baseline control differs from frozen cache')
        checked.append({'cohort':cohort,'all_terminal_fields_identical':True})
    write(out/'baseline-controls-private.json',checked);policy=candidate_agent(client);records=[]
    for entry in entries:
        cohort,source,position,_=entry;name=f"{position['seed']}-{position['step']}-{position['seat']}";stored=cached[(cohort,name)]
        candidate=rollout(environment(entry),position['seat'],position['seed'],policy,30000)
        row={k:stored[k] for k in ('seed','step','seat','initial_rank','baseline')}
        row.update(cohort=cohort,candidate=candidate,baseline_reused=True,policy_sha256=digest(protocol))
        write(out/'rollouts'/cohort/(name+'.json'),row);records.append(row)
        status={'status':'candidate_complete_match_continuations','complete':len(records),'total':83,
            'baseline_reused':83,'baseline_controls_completed':2}
        write(out/'status-private.json',status);print(json.dumps(status),flush=True)
    result=summarize(records)
    for path,expected in protocol['source_sha256'].items():
        if checksum(path)!=expected:raise ValueError('Frozen source/cache changed during development')
    if bundle_identity(parent)!=identity:raise ValueError('Formal model files changed')
    summary=dict(result,status='fourth_scope_consumed_development_complete',baseline_controls=checked,
        static_path_ablation=ablation,
        baseline_rollouts_reused=83,new_candidate_rollouts=83,baseline_control_rollouts=2,
        formal_weights_unchanged=True,independent_confirmation=False,deployed=False,raw_publication=False)
    write(out/'summary-private.json',summary);write(out/'status-private.json',{'status':summary['status'],
        'eligible_for_new_confirmation':summary['eligible_for_new_confirmation']});return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Full continuation development requires a scheduled GPU allocation')
    from tools.prepare_native_context_replay import allocation_metadata
    description=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True)
    print(json.dumps({'node':socket.gethostname(),'allocation':allocation_metadata(description,socket.gethostname())}),flush=True)
    print(json.dumps(evaluate(json.loads(args.config.read_text())),indent=2),flush=True)


if __name__=='__main__':main()
