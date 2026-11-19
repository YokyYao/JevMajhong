"""TRAIN-only ablation of escape and ordinary-action guards for fourth-place pursuit."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def validate_config(config):
    if (config.get('schema')!='endgame-fourth-risk-audit-v1'
            or config.get('out')!='runs/endgame-fourth-chase-risk-audit-v1'
            or config.get('fourth_training_positions')!=149 or config.get('ordinary_fourth_positions')!=55
            or config.get('win_fourth_positions')!=94
            or config.get('arms')!=['original_safe','no_escape_floor','top2_no_floor','top2_legal_ordinary']
            or any(config.get(k)!=0 for k in ('optimizer_updates','new_model_queries','new_simulations'))
            or any(config.get(k) is not False for k in ('validation_data_used','fit_registered','independent_confirmation','raw_publication'))):
        raise ValueError('Only the complete149 TRAIN-only risk audit is registered')


def audit(config):
    validate_config(config)
    from jevmahjong.endgame.prepare import read_jsonl,checksum
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.native_terminal import is_native_terminal
    from jevmahjong.endgame.public_policy import action_keys
    from jevmahjong.endgame.reference import escape_supported_keys,greedy_improvement_target,greedy_win_target
    from jevmahjong.endgame.fourth_chase import top2_chase_target,fourth_chase_action_keys
    from tools.audit_endgame_training_label_stability import outcome_arrays
    from desktop.jev_sidecar.unified_policy import identity
    run,out=Path(config['source_data_run']),Path(config['out'])
    if any((out/name).exists() for name in ('protocol-private.json','summary-private.json')):
        raise ValueError('Inspect completed or interrupted audit instead of silently restarting')
    source_config=json.loads((run/'config-private.json').read_text());proof=json.loads((run/'preparation-private.json').read_text())
    if checksum(run/'train.jsonl')!=proof['files']['train']['sha256']:raise ValueError('Source training data changed')
    parent=bundle_identity(source_config['checkpoint']);released=identity()
    if parent['identity_sha256'] not in (released['identity_sha256'],released['source_identity_sha256']):
        raise ValueError('Source is not formal v8')
    reference=json.loads((run/'reference-targets-private.json').read_text())
    if reference['identity_sha256']!=parent['identity_sha256']:raise ValueError('Reference model differs')
    rows=[r for r in read_jsonl(run/'train.jsonl') if r['questions']['action']['src'] in
        ('endgame_terminal_fourth','endgame_terminal_fourth_improve','endgame_terminal_win_fourth')]
    if len(rows)!=149 or sum(r['questions']['action']['src']=='endgame_terminal_win_fourth' for r in rows)!=94:
        raise ValueError('Complete registered149 training fourth-place menus changed')
    paths=set(ROOT.glob('jevmahjong/**/*.py'))|set(ROOT.glob('desktop/jev_sidecar/**/*.py'))
    paths.update(ROOT/name for name in ('tools/audit_endgame_fourth_risk.py','tools/audit_endgame_training_label_stability.py',
        'configs/training/endgame-fourth-risk-audit-v1.json','tests/core/test_endgame_fourth_chase.py'))
    paths.update(run/name for name in ('train.jsonl','config-private.json','preparation-private.json','reference-targets-private.json'))
    raw=[]
    for row in rows:
        meta=row['_meta'];provenance=meta['terminal_label_provenance']
        root=Path(source_config['native_terminal_label_root']) if is_native_terminal(meta) else run
        path=root/'labels'/(meta['id'].split(':',1)[1]+'.json')
        if checksum(path)!=provenance['file_sha256']:raise ValueError('Original admitted label changed')
        paths.add(path);raw.append((row,json.loads(path.read_text())))
    protocol={'configuration':config,'parent_identity':parent,'training_ids':[r['_meta']['id'] for r in rows],
        'source_sha256':{str(p.resolve()):checksum(p) for p in sorted(paths)}}
    write(out/'protocol-private.json',protocol);cases=[]
    for row,label in raw:
        meta,q=row['_meta'],row['questions']['action'];rid=meta['id'];keys=list(q['criteria']);seat=label['source']['seat']
        if meta['split']!='train' or meta['source_initial_placement']!=4:raise ValueError('Nontraining/nonfourth position')
        utilities,escape=outcome_arrays(label,keys,seat,meta['terminal_label_provenance']['worlds_per_action'],parent['identity_sha256'])
        mean=lambda v:sum(v)/len(v)
        u={k:mean(v) for k,v in utilities.items()};e={k:mean(v) for k,v in escape.items()}
        t={k:mean([o['ranks'][seat]<=2 for o in outcomes]) for k,outcomes in zip(keys,label['outcomes'])}
        if any(abs(u[k]-meta['expected_terminal_utilities'][k])>1e-12 or abs(e[k]-meta['terminal_escape4_probabilities'][k])>1e-12 for k in keys):
            raise ValueError('Original utility/escape annotations differ from full native outcomes')
        prior=reference['rows'][rid]['probabilities'];eligible=action_keys(row);safe=escape_supported_keys(prior,e,eligible)
        win=q['src']=='endgame_terminal_win_fourth'
        target_old=(greedy_win_target(q['criteria'],u,safe) if win else greedy_improvement_target(prior,u,safe))
        target_free=(greedy_win_target(q['criteria'],u,eligible) if win else greedy_improvement_target(prior,u,eligible))
        targets={'original_safe':target_old,'no_escape_floor':target_free,
            'top2_no_floor':top2_chase_target(prior,t,e,eligible),
            'top2_legal_ordinary':top2_chase_target(prior,t,e,fourth_chase_action_keys(row))}
        if any(abs(target_old[k]-q['target'][k])>1e-12 for k in keys):raise ValueError('Old target does not reproduce')
        choices={name:max(target,key=target.get) for name,target in targets.items()}
        base=choices['original_safe'];new=choices['top2_legal_ordinary'];original=max(eligible,key=prior.get)
        cases.append({'id':rid,'kind':'win' if win else 'ordinary','worlds':label['worlds_per_action'],
            'arms':{name:{'choice':key,'top2':t[key],'fourth':1-e[key],'old_utility':u[key],
                'action_type':q['criteria'][key]['type']} for name,key in choices.items()},
            'floor_excludes_higher_top2':any(t[k]>t[base]+1e-12 for k in eligible if k not in safe),
            'ordinary_guard_excludes_higher_top2':not win and any(t[k]>t[choices['top2_no_floor']]+1e-12 for k in keys if k not in eligible),
            'new_strict_top2_improvement_vs_original':t[new]>t[original]+1e-12})
    groups={'all':cases,'ordinary':[c for c in cases if c['kind']=='ordinary'],'win':[c for c in cases if c['kind']=='win']}
    summaries={}
    for group,values in groups.items():
        summaries[group]={'positions':len(values),'world_counts':dict(Counter(c['worlds'] for c in values)),
            'floor_excludes_higher_top2_positions':sum(c['floor_excludes_higher_top2'] for c in values),
            'ordinary_guard_excludes_higher_top2_positions':sum(c['ordinary_guard_excludes_higher_top2'] for c in values),
            'new_strict_top2_improvements':sum(c['new_strict_top2_improvement_vs_original'] for c in values),
            'arms':{arm:{metric:sum(c['arms'][arm][metric] for c in values)/len(values) for metric in ('top2','fourth','old_utility')}
                    for arm in config['arms']},
            'choice_changes_from_old':{arm:sum(c['arms'][arm]['choice']!=c['arms']['original_safe']['choice'] for c in values)
                    for arm in config['arms']}}
    for path,expected in protocol['source_sha256'].items():
        if checksum(path)!=expected:raise ValueError('Read-only source changed during audit')
    if bundle_identity(source_config['checkpoint'])!=parent:raise ValueError('Formal v8 changed')
    write(out/'cases-private.json',cases)
    summary={'status':'complete_training_fourth_risk_audit','cohorts':summaries,'old_targets_and_ledgers_reproduced':True,
        'optimizer_updates':0,'new_model_queries':0,'new_simulations':0,'validation_data_used':False,
        'independent_confirmation':False,'raw_publication':False,'finite_labels_not_actual_strength':True}
    write(out/'summary-private.json',summary);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Fourth-risk data processing requires a scheduled GPU allocation')
    from tools.prepare_native_context_replay import allocation_metadata
    description=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True)
    print(json.dumps({'node':socket.gethostname(),'allocation':allocation_metadata(description,socket.gethostname())}),flush=True)
    print(json.dumps(audit(json.loads(args.config.read_text())),indent=2),flush=True)


if __name__=='__main__':main()
