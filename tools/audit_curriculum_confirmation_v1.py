"""Independently reconstruct completed512-game final evidence after outcomes.

Reserved initial-wall resets happen only after both complete result matrices
and their actual frozen registration have been checked. No model is trained.
"""
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only,write
from tools.curriculum_confirmation_runtime_v1 import require_header,verify,cells
from tools.curriculum_confirmation_stats_v1 import matrix,compare
from tools.curriculum_confirmation_contract_v1 import assess_statistics
from tools.curriculum_confirmation_native_v1 import audit_matches


def require_complete_outcomes(reg,opened,result,records,registration_sha):
    require_header(reg)
    if (result['status']!='actual_fixed512_confirmation_complete_pending_independent_audit'
        or result['registered_sha256']!=registration_sha or opened['registered_sha256']!=registration_sha
        or result['actual_complete_games']!=512 or result['optimizer_updates']!=0
        or not result['sole_candidate_unchanged'] or not result['independent_postrun_audit_still_required']
        or opened['candidate_identity']!=reg['candidate_model']['identity_sha256']
        or opened['reference_identity']!=reg['reference_model']['identity_sha256']
        or opened['seed_start']!=2026300000 or opened['seed_groups']!=32
        or opened['scheduled_matches_per_policy']!=256 or not opened['no_optimizer_updates']
        or result['models']!={role:reg[role+'_model'] for role in ('candidate','reference')}
        or result['policy_identities']!={role:reg[role+'_policy_identity'] for role in ('candidate','reference')}
        or set(records)!={'candidate','reference'}):
        raise ValueError('Both actual complete final outcome matrices must precede any reserved audit reset')
    for rows in records.values():matrix(rows,2026300000)


def initial_prefix(seed):
    from riichienv import RiichiEnv,GameRule
    from jevmahjong.core.seeds import engine_seed
    env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(seed,'spaced-v1'))
    env.reset();return copy.deepcopy(env.mjai_log)


def verify_seed_prefixes(records,prefix_factory):
    # Called only after require_complete_outcomes in the real entry point.
    seeds=sorted({row['seed'] for rows in records.values() for row in rows})
    prefixes={seed:prefix_factory(seed) for seed in seeds};checked=0
    for rows in records.values():
        for row in rows:
            expected=prefixes[row['seed']]
            if not expected or row['mjai_log'][:len(expected)]!=expected:
                raise ValueError('Actual initial wall/log prefix differs from the declared seed and native rules')
            checked+=1
    return dict(initial_native_prefixes_reconstructed=len(prefixes),actual_match_prefixes_checked=checked)


def audit(registered,out):
    gpu_only();reg=read(registered);require_header(reg);registration_sha=sha(registered)
    verified=verify(reg,registered);directory=Path(reg['out']).resolve()
    opened_path=directory/'opened-private.json';result_path=directory/'comparison-private.json'
    opened=read(opened_path);result=read(result_path)
    records={role:[json.loads(line) for line in (directory/role/'matches-private.jsonl').open()]
        for role in ('reference','candidate')}
    require_complete_outcomes(reg,opened,result,records,registration_sha)
    sys.path.insert(0,reg['runtime']['source']);sys.path.insert(0,reg['runtime']['upstream_source'])
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.evaluation.unified_contract import bundle
    files={str(registered.resolve()):registration_sha};summaries={};native={}
    for role,rows in records.items():
        folder=directory/role;protocol=read(folder/'protocol.json');stored=read(folder/'summary.json')
        if (protocol['model']!=reg[role+'_model'] or protocol['policy']!=reg[role+'_policy']
            or protocol['policy_identity']!=reg[role+'_policy_identity']
            or protocol['scheduled_cells']!=[list(value) for value in cells(reg['criteria'])]
            or protocol['registered_sha256']!=registration_sha or protocol['game_mode']!=reg['game_mode']
            or protocol['rule']!=reg['rule'] or protocol['seed_namespace']!=reg['seed_namespace']):
            raise ValueError('Actual final role/policy/scheduling protocol differs')
        native[role]=audit_matches(rows,2026300000,settlement_ledger,hand_dealin_count)
        summaries[role]=aggregate_complete_matches(rows)
        if (any(stored[key]!=value for key,value in summaries[role].items())
            or stored['parameter_identity']!=verified[role]['identity_sha256']
            or stored['policy_identity']!=reg[role+'_policy_identity']
            or result['summaries'][role]!=summaries[role] or result['native_checks'][role]!=native[role]):
            raise ValueError('Actual full256 native aggregation differs from the claimed outcome')
        for name in ('protocol.json','summary.json','matches-private.jsonl'):
            path=(folder/name).resolve();files[str(path)]=sha(path)
    prefixes=verify_seed_prefixes(records,initial_prefix)
    anchor=read(directory/'reference-development-anchor-private.json')
    expected=[json.loads(line) for line in Path(reg['reference_development_anchor']).open()]
    if anchor!=next(row for row in expected if row['profile']=='efficiency' and row['seat']==0):
        raise ValueError('Original v8 consumed-development anchor differs')
    comparison=compare(records['candidate'],records['reference'],2026300000)
    arithmetic=assess_statistics(comparison,reg['criteria'])
    if comparison!=result['comparison'] or arithmetic!=result['acceptance_arithmetic']:
        raise ValueError('Actual32-seed arithmetic or secondary disclosure differs')
    verify(reg,registered)
    if sha(registered)!=registration_sha or any(bundle(reg[role+'_model']['directory'])!=reg[role+'_model'] for role in records):
        raise ValueError('Final models or registration changed during independent evidence audit')
    for path in (Path(__file__),opened_path,result_path,directory/'reference-development-anchor-private.json'):
        files[str(path.resolve())]=sha(path)
    evidence=dict(status='actual_final512_model_policy_seed_and_financial_evidence_independently_audited',
        models={role:reg[role+'_model'] for role in ('candidate','reference')},
        policy_identities={role:reg[role+'_policy_identity'] for role in ('candidate','reference')},
        actual_complete_games=512,independent_seed_groups=32,native_checks=native,seed_prefix_checks=prefixes,
        summaries=summaries,comparison=comparison,acceptance_arithmetic=arithmetic,
        primary_registered_objective_evidence_supported=arithmetic['primary_arithmetic_conditions_met'],
        claim_scope='registered_top2_and_net_points_under_fixed_efficiency_calling_opponents',
        real_human_rank_or_all_metrics_superiority_not_established=True,
        optimizer_updates=0,models_and_source_binding_unchanged=True,task_completion_or_release_authorized=False)
    write(out,evidence);write(out.parent/'confirmation-evidence-binding-private.json',dict(files=files))
    print(json.dumps(dict(status=evidence['status'],actual_complete_games=512,
        primary_registered_objective_evidence_supported=evidence['primary_registered_objective_evidence_supported'],
        task_completion_or_release_authorized=False)),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.registered.resolve(),a.out.resolve())
