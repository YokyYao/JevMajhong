"""Execute the sole sealed candidate/v8 fixed512-match final comparison.

No final registration currently exists. Training/statistical plan files are
rejected. This program cannot create or replace the missing source admission.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only,write
from tools.curriculum_confirmation_runtime_v1 import require_header,verify,cells,require_output_isolation
from tools.curriculum_confirmation_native_v1 import FrozenReferenceEventAdapter,audit_matches
from tools.curriculum_confirmation_stats_v1 import compare
from tools.curriculum_confirmation_contract_v1 import assess_statistics
from tools.play_curriculum_placement_v1 import play
from tools.evaluate_curriculum_riichi_class_route_v2 import status_write


def run_cells(agent,scheduled,play_game,emit,progress):
    """One actual complete result for every predeclared cell; count-only progress."""
    records=[]
    for seed,profile,seat in scheduled:
        row=play_game(agent,seed,seat,profile)
        if ((row['seed'],row['profile'],row['seat'])!=(seed,profile,seat)
            or row['completed'] is not True or row['ledger_passed'] is not True):
            raise ValueError('Actual complete game must match its original scheduled cell')
        emit(row);records.append(row);progress(len(records),len(scheduled))
    return records


def run(registered):
    gpu_only();reg=read(registered);require_header(reg);initial_registration_sha=sha(registered)
    verified=verify(reg,registered);runtime=reg['runtime']
    os.environ['JEV_BASE_MODEL_DIR']=verified['placement']['training_config']['base_load_path']
    sys.path.insert(0,runtime['source']);sys.path.insert(0,runtime['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.evaluation.agents import PolicyAgent
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from tools.curriculum_placement_full_win_policy_v1 import agent as placement_agent
    directory=Path(reg['out']).resolve()
    require_output_isolation(directory,[verified[role]['directory'] for role in ('reference','candidate')]+
        [ROOT/verified['placement']['curriculum']['out'],Path(runtime['source']),Path(runtime['upstream_source'])])
    directory.mkdir(parents=True,exist_ok=False)
    scheduled=cells(reg['criteria']);results={};summaries={};native={};opened=False;started=time.monotonic()
    for role in reg['role_order']:
        verify(reg,registered)
        model=load_model(verified[role]['directory']);client=UnifiedPolicyClient(model)
        agent=FrozenReferenceEventAdapter(PolicyAgent(client)) if role=='reference' else placement_agent(client)
        if role=='reference':
            anchors=[json.loads(line) for line in Path(reg['reference_development_anchor']).open()]
            expected=next(r for r in anchors if r['profile']=='efficiency' and r['seat']==0)
            if reg['criteria']['seed_start']<=expected['seed']<reg['criteria']['seed_start']+32:
                raise ValueError('A reserved outcome cannot serve as a development anchor')
            actual=play(agent,expected['seed'],expected['seat'],expected['profile'])
            write(directory/'reference-development-anchor-private.json',actual)
            if actual!=expected:raise ValueError('Original v8 development event/outcome reproduction failed before opening')
        if not opened:
            if sha(registered)!=initial_registration_sha:raise ValueError('Sole final registration changed before opening')
            write(directory/'opened-private.json',dict(registered_sha256=initial_registration_sha,
                candidate_identity=verified['candidate']['identity_sha256'],reference_identity=verified['reference']['identity_sha256'],
                seed_start=2026300000,seed_groups=32,scheduled_matches_per_policy=256,
                node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],no_optimizer_updates=True))
            opened=True
        folder=directory/role;folder.mkdir(exist_ok=False)
        write(folder/'protocol.json',dict(model=verified[role],policy=verified[role+'_policy'],
            policy_identity=reg[role+'_policy_identity'],scheduled_cells=scheduled,
            kind='fixed_independent_final_not_development',game_mode=reg['game_mode'],rule=reg['rule'],
            seed_namespace=reg['seed_namespace'],registered_sha256=initial_registration_sha))
        with (folder/'matches-private.jsonl').open('x') as stream:
            def emit(row):stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush()
            def progress(done,total):
                status_write(directory/'status.json',dict(status='running',role=role,complete=done,total=total,
                    elapsed_seconds=time.monotonic()-started,partial_strength_summary_available=False))
            records=run_cells(agent,scheduled,play,emit,progress)
        native[role]=audit_matches(records,2026300000,settlement_ledger,hand_dealin_count)
        summaries[role]=aggregate_complete_matches(records);results[role]=records
        if bundle(verified[role]['directory'])!=verified[role]:raise ValueError('Confirmation changed model files')
        write(folder/'summary.json',dict(**summaries[role],parameter_identity=verified[role]['identity_sha256'],
            policy_identity=reg[role+'_policy_identity'],kind='independent_final'))
        del agent,client,model;gc.collect();torch.cuda.empty_cache()
    verify(reg,registered)
    if sha(registered)!=initial_registration_sha:raise ValueError('Final registration changed during outcomes')
    comparison=compare(results['candidate'],results['reference'],2026300000)
    arithmetic=assess_statistics(comparison,reg['criteria'])
    result=dict(status='actual_fixed512_confirmation_complete_pending_independent_audit',
        registered_sha256=initial_registration_sha,models=dict(candidate=verified['candidate'],reference=verified['reference']),
        policy_identities={role:reg[role+'_policy_identity'] for role in reg['role_order']},
        summaries=summaries,comparison=comparison,acceptance_arithmetic=arithmetic,native_checks=native,
        actual_complete_games=512,optimizer_updates=0,sole_candidate_unchanged=True,
        task_completion_or_release_authorized=False,independent_postrun_audit_still_required=True)
    write(directory/'comparison-private.json',result)
    status_write(directory/'status.json',dict(status=result['status'],actual_complete_games=512))
    print(json.dumps(dict(status=result['status'],actual_complete_games=512,
        primary_arithmetic_conditions_met=arithmetic['primary_arithmetic_conditions_met'],
        task_completion_or_release_authorized=False)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registered',type=Path,required=True)
    run(p.parse_args().registered.resolve())
