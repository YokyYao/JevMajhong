"""I/O-only v2: preserve prior16 exact games and completed fixed-grid prefix."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.curriculum_complete_legal_class_policy_v1 import CompleteLegalClassPolicyAgent, METHOD
from tools.curriculum_riichi_class_policy_v1 import RiichiClassPolicyAgent
from tools.evaluate_curriculum_calls_classroute_v1 import policy_contract
from tools.evaluate_curriculum_riichi_class_route_v2 import status_write
from tools.evaluate_curriculum import play


def evaluate(out, audit_job):
    gpu_only()
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    parent = ROOT/'runs/curriculum-calls-classroute-v1'; registration = parent/'fit-plan/registered.json'
    reg = read(registration); c = reg['curriculum']; fit = ROOT/c['out']
    audit_exit = ROOT/'logs'/f'curriculum-calls388-audit-exit-{audit_job}.json'
    prior = ROOT/'runs/curriculum-complete-legal-class-v1/native64-v1'
    prior_terminal = ROOT/'runs/curriculum-complete-legal-class-v1/native64-v1-terminal-scheduler-private.txt'
    prior_exit = ROOT/'logs/curriculum-complete-legal-native64-exit-318059.json'
    raw_failure = prior_terminal.read_text(); failed = read(prior_exit)
    if (str(failed['job_id']) != '318059' or failed['exit_code'] != 1
            or 'JobState=FAILED' not in raw_failure or 'ExitCode=1:0' not in raw_failure
            or 'cluster/curriculum-complete-legal-class-native-v1.slurm' not in raw_failure):
        raise ValueError('Actual unchanged v1 reporting-I/O failure required')
    terminal = read(prior/'parent-audit-terminal-private.json')['raw']
    scheduler_exit(read(audit_exit), terminal, parent/'audit.slurm')
    evidence = read(parent/'final-evidence-private.json'); done = read(fit/'completion.json')
    if (evidence['status'] != 'stage2_calls_classroute_final_evidence_audit_passed'
            or evidence['optimizer_steps'] != 388 or evidence['real_records_seen'] != 98044
            or done['status'] != 'stage2_calls_classroute_fit_and_curves_complete'
            or read(parent/'audit-execution-private.json')['status'] != 'actual_calls388_independent_full_evidence_audited'):
        raise ValueError('Actual completed final388 and independent evidence audit required')
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    if package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']: raise ValueError('Pinned runtime changed')
    path = ROOT/'configs/diagnostics/curriculum-complete-legal-class-native-v1.json'; config = read(path)
    if (config['method'] != METHOD or config['step'] != 388 or config['groups'] != 8
            or config['old_route_exact_cached_reproductions'] != 16 or config['new_route_complete_matches'] != 64
            or config['automatic_fit_or_release'] is not False): raise ValueError('Preregistered same-parameter comparison required')
    checkpoint = fit/'training-run/checkpoints/step-000388'; identity = bundle(checkpoint)
    oldroot = fit/'evaluations/step-000388'; oldprotocol = read(oldroot/'protocol.json'); oldsummary = read(oldroot/'summary.json')
    oldgames = [json.loads(line) for line in (oldroot/'matches-private.jsonl').open()]
    oldspec, oldpolicy = policy_contract(identity, reg)
    if (oldprotocol['checkpoint'] != identity or oldprotocol['policy'] != oldspec or oldprotocol['policy_identity'] != oldpolicy
            or oldsummary['checkpoint_sha256'] != identity['identity_sha256'] or len(oldgames) != 64):
        raise ValueError('Final parameter bundle and audited original route cache differ')
    grid = {(seed, profile, seat) for seed in range(c['curve_seed_start'], c['curve_seed_start']+8)
            for profile in c['opponent_profiles'] for seat in range(4)}
    def key(row): return row['seed'], row['profile'], row['seat']
    if len(grid) != 64 or {key(r) for r in oldgames} != grid: raise ValueError('Exact consumed64 development grid required')
    preflight = ROOT/'runs/curriculum-complete-legal-class-v1/preflight-v1'; pre = read(preflight/'summary-private.json')
    if (pre['status'] != 'complete_legal_fixed_RI_menu_and_native_interface_preflight_passed_not_fit_admitted'
            or pre['all_teacher_menus_checked'] != 82419): raise ValueError('Complete native-interface preflight required')
    files = dict(reg['inputs']); active = read(parent/'fit-launch-binding-private.json')['files']
    for p, h in read(preflight/'completion-binding-private.json')['files'].items():
        if p in files and files[p] != h: raise ValueError('Conflicting bound source identity')
        files[p] = h
    paths = [Path(__file__), path, registration, parent/'final-evidence-private.json', parent/'audit-execution-private.json',
        parent/'fit-terminal-scheduler-private.json', fit/'completion.json', audit_exit,
        ROOT/'tools/curriculum_complete_legal_class_policy_v1.py', ROOT/'tools/curriculum_riichi_class_policy_v1.py',
        ROOT/'tools/evaluate_curriculum.py', ROOT/'tools/evaluate_curriculum_calls_classroute_v1.py',
        ROOT/'cluster/curriculum-complete-legal-class-native-v1.slurm',
        ROOT/'docs/experiments/plans/curriculum-complete-legal-class-native-20261123.md',
        ROOT/'tools/evaluate_curriculum_complete_legal_class_v1.py',
        ROOT/'cluster/curriculum-complete-legal-class-native-v2.slurm',
        ROOT/'docs/experiments/plans/curriculum-complete-legal-class-native-v2-20261123.md',
        prior_terminal, prior_exit, prior/'registered-private.json', prior/'parent-audit-terminal-private.json',
        prior/'original-route-reproduction-private.jsonl', prior/'matches-private.jsonl', prior/'status-private.json',
        preflight/'summary-private.json', preflight/'completion-binding-private.json']
    paths.extend(oldroot/name for name in ('protocol.json', 'summary.json', 'matches-private.jsonl'))
    paths.extend(checkpoint/name for name in identity['files'])
    for p in paths: files[str(p.resolve())] = sha(p)
    if any(sha(p) != h for p, h in {**files, **active}.items()): raise ValueError('Frozen comparison inputs changed')
    spec = dict(method=METHOD, route_sha256=sha(ROOT/'tools/curriculum_complete_legal_class_policy_v1.py'),
        RI_class_rule_sha256=sha(ROOT/'tools/curriculum_riichi_class_policy_v1.py'), config_sha256=sha(path),
        parameter_identity=identity['identity_sha256'], runtime_source_sha256=reg['source_sha256'],
        forced_legal_wins=True, guard_only_shadow=True, single_forward_per_nonshortcut=True)
    policy_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if policy_id == oldpolicy: raise ValueError('Changed route must have a distinct policy identity')
    previous = read(prior/'registered-private.json')
    if previous['parameter_bundle'] != identity or previous['policy'] != spec or previous['policy_identity'] != policy_id:
        raise ValueError('I/O repair cannot change the original numerical/policy contract')
    for p, h in previous['files'].items():
        if p in files and files[p] != h: raise ValueError('Preserved v1 input identity conflicts with repair')
        files[p] = h
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Preserved v1 input changed')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'parent-audit-terminal-private.json', dict(raw=terminal, exit=read(audit_exit)))
    write(out/'registered-private.json', dict(files=files, active_fit_files=active, parent=evidence['stage2_advancement'],
        config=config, parameter_bundle=identity, original_policy_identity=oldpolicy, policy=spec, policy_identity=policy_id,
        original_quality_grade_not_overridden=True, comparison_not_parent_or_fit_admission=True))
    baseline = [r for r in oldgames if r['seed'] < c['curve_seed_start']+2]
    preserved_baseline = [json.loads(line) for line in (prior/'original-route-reproduction-private.jsonl').open()]
    if preserved_baseline != baseline or len(baseline) != 16:
        raise ValueError('All actual prior16 original-route full event reproductions required')
    with (out/'original-route-reproduction-private.jsonl').open('x') as stream:
        for actual in preserved_baseline:
            stream.write(json.dumps(actual)+'\n'); stream.flush()
    preserved = [json.loads(line) for line in (prior/'matches-private.jsonl').open()]
    ordered_grid = [(seed, profile, seat) for seed in range(c['curve_seed_start'], c['curve_seed_start']+8)
        for profile in c['opponent_profiles'] for seat in range(4)]
    if len(preserved) != 2 or [key(r) for r in preserved] != ordered_grid[:2]:
        raise ValueError('Exact unselected completed two-game prefix required')
    client = UnifiedPolicyClient(load_model(checkpoint))
    agent = CompleteLegalClassPolicyAgent(client); games = list(preserved); started = time.monotonic()
    with (out/'matches-private.jsonl').open('x') as stream:
        for row in preserved: stream.write(json.dumps(row)+'\n')
        stream.flush()
        for seed in range(c['curve_seed_start'], c['curve_seed_start']+8):
            for profile in c['opponent_profiles']:
                for seat in range(4):
                    if (seed, profile, seat) in ordered_grid[:2]: continue
                    row = play(agent, seed, seat, profile); games.append(row)
                    stream.write(json.dumps(row)+'\n'); stream.flush()
                    status_write(out/'status.json', dict(status='native_same_parameter_route_comparison_running',
                        completed=len(games), total=64, seconds=time.monotonic()-started))
    if {key(r) for r in games} != grid or len(games) != 64: raise ValueError('Changed grid incomplete')
    for row in games:
        ledger = settlement_ledger(row['mjai_log'], row['final_scores']); seat = row['seat']
        if (not row['completed'] or not row['ledger_passed'] or ledger['hands'] != row['hands']
                or row['rank'] != sorted(range(4), key=lambda p: (-row['final_scores'][p], p)).index(seat)+1
                or row['net_points'] != row['final_scores'][seat]-25000
                or row['dealin_hands'] != hand_dealin_count(row['mjai_log'], seat)
                or row['wins'] != sum(e['type'] == 'hora' and e.get('actor') == seat for e in row['mjai_log'])):
            raise ValueError('Changed complete native game accounting differs')
    comparisons = {}; oldmap = {key(r): r for r in oldgames}
    for field, fn in [('net_points', lambda r: r['net_points']), ('top2', lambda r: float(r['rank'] <= 2)),
        ('first', lambda r: float(r['rank'] == 1)), ('fourth', lambda r: float(r['rank'] == 4)),
        ('hand_win', lambda r: r['wins']/r['hands']), ('dealin', lambda r: r['dealin_hands']/r['hands'])]:
        clusters = [statistics.mean(fn(r)-fn(oldmap[key(r)]) for r in games if r['seed'] == seed)
            for seed in range(c['curve_seed_start'], c['curve_seed_start']+8)]
        mean = statistics.mean(clusters); radius = 2.3646242510102993*statistics.stdev(clusters)/math.sqrt(8)
        comparisons[field] = dict(gain=mean, ci95=[mean-radius, mean+radius], seed_groups=8, consumed_development_only=True)
    if bundle(checkpoint) != identity or any(sha(p) != h for p, h in {**files, **active}.items()):
        raise ValueError('Parameters/source/active fit changed during diagnostic')
    result = dict(status='completed_calls388_same_parameters_complete_legal_route_native64_compared',
        checkpoint_identity=identity['identity_sha256'], original_policy_identity=oldpolicy, proposed_policy_identity=policy_id,
        actual_original_route_exact_games_reproduced=16, original_summary=aggregate_complete_matches(oldgames),
        actual_original_route16_reproductions_preserved_from_failed_v1=True,
        actual_completed_proposed_prefix_games_preserved_from_v1=2, new_proposed_games=62,
        proposed_summary=aggregate_complete_matches(games), comparisons_against_original_route=comparisons,
        diagnostics=dict(agent.diagnostics), diagnostics_scope='new62_games_only_prior2_shadow_counters_not_saved',
        proposed_operation_counts=dict(Counter({kind: sum(r['choices'].get(kind, 0) for r in games)
            for kind in ('chi', 'pon', 'ankan', 'daiminkan', 'kakan', 'reach')})),
        unchanged_original_parent_quality_grade=evidence['stage2_advancement'],
        independent_strength_or_v8_parity_proven=False, final_confirmation_seeds_opened=False,
        optimizer_updates=0, fit_or_release_admitted=False, active_fit_dependencies_unchanged=len(active), raw_events_private=True)
    write(out/'summary-private.json', result); write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active))
    status_write(out/'status.json', dict(status='complete', games=64))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--audit-job', type=int, required=True); a = p.parse_args(); evaluate(a.out.resolve(), a.audit_job)
