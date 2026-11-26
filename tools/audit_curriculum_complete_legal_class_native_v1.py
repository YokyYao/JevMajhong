"""Independent cached64-game execution-route accounting and policy identity audit."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write


def audit(source, out, terminal, exit_path):
    gpu_only()
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    scheduler_exit(read(exit_path), terminal.read_text(), ROOT/'cluster/curriculum-complete-legal-class-native-v2.slurm')
    reg = read(source/'registered-private.json'); result = read(source/'summary-private.json')
    files = dict(read(source/'completion-binding-private.json')['files'])
    for p in (Path(__file__), source/'registered-private.json', source/'summary-private.json',
              source/'completion-binding-private.json', source/'matches-private.jsonl',
              source/'original-route-reproduction-private.jsonl', source/'status.json', terminal, exit_path,
              ROOT/'cluster/curriculum-complete-legal-class-native-audit-v1.slurm',
              ROOT/'docs/experiments/plans/curriculum-complete-legal-class-native-audit-20261123.md'):
        files[str(p.resolve())] = sha(p)
    active = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files']
    if any(sha(p) != h for p, h in {**files, **active}.items()): raise ValueError('Completed comparison binding changed')
    if (result['status'] != 'completed_calls388_same_parameters_complete_legal_route_native64_compared'
            or read(source/'status.json') != dict(status='complete', games=64)
            or reg['original_quality_grade_not_overridden'] is not True):
        raise ValueError('Actually completed unchanged64-game diagnostic required')
    parent = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-plan/registered.json')
    sys.path.insert(0, parent['source']); sys.path.insert(0, parent['training_config']['upstream_source'])
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from tools.curriculum_complete_legal_class_policy_v1 import METHOD
    fit = ROOT/parent['curriculum']['out']; identity = bundle(fit/'training-run/checkpoints/step-000388')
    from tools.evaluate_curriculum_calls_classroute_v1 import policy_contract
    old_spec, old_policy_id = policy_contract(identity, parent)
    old_protocol = read(fit/'evaluations/step-000388/protocol.json')
    if (old_protocol['policy'] != old_spec or old_protocol['policy_identity'] != old_policy_id
            or result['original_policy_identity'] != old_policy_id or reg['original_policy_identity'] != old_policy_id):
        raise ValueError('Original guarded policy identity differs from actual audited cache')
    expected_policy = dict(method=METHOD, route_sha256=sha(ROOT/'tools/curriculum_complete_legal_class_policy_v1.py'),
        RI_class_rule_sha256=sha(ROOT/'tools/curriculum_riichi_class_policy_v1.py'),
        config_sha256=sha(ROOT/'configs/diagnostics/curriculum-complete-legal-class-native-v1.json'),
        parameter_identity=identity['identity_sha256'], runtime_source_sha256=parent['source_sha256'],
        forced_legal_wins=True, guard_only_shadow=True, single_forward_per_nonshortcut=True)
    policy_id = hashlib.sha256(json.dumps(expected_policy, sort_keys=True).encode()).hexdigest()
    if (reg['parameter_bundle'] != identity or result['checkpoint_identity'] != identity['identity_sha256']
            or reg['policy'] != expected_policy or reg['policy_identity'] != policy_id
            or result['proposed_policy_identity'] != policy_id): raise ValueError('Complete parameter/selector identity differs')
    def rows(path): return [json.loads(line) for line in path.open()]
    original = rows(fit/'evaluations/step-000388/matches-private.jsonl'); proposed = rows(source/'matches-private.jsonl')
    prior = ROOT/'runs/curriculum-complete-legal-class-v1/native64-v1'
    old16 = rows(source/'original-route-reproduction-private.jsonl')
    if (old16 != rows(prior/'original-route-reproduction-private.jsonl')
            or old16 != [r for r in original if r['seed'] < parent['curriculum']['curve_seed_start']+2]
            or len(old16) != 16 or proposed[:2] != rows(prior/'matches-private.jsonl')):
        raise ValueError('Actual16 baseline reproductions or exact completed two-game prefix changed')
    key = lambda r: (r['seed'], r['profile'], r['seat'])
    grid = {(s, p, seat) for s in range(parent['curriculum']['curve_seed_start'], parent['curriculum']['curve_seed_start']+8)
        for p in parent['curriculum']['opponent_profiles'] for seat in range(4)}
    summaries = {}; checked = 0
    for name, games in (('original', original), ('proposed', proposed)):
        if len(games) != 64 or {key(r) for r in games} != grid: raise ValueError('Exact complete64-game grid required')
        for r in games:
            seat = r['seat']; log = r['mjai_log']; ledger = settlement_ledger(log, r['final_scores'])
            rank = sorted(range(4), key=lambda p: (-r['final_scores'][p], p)).index(seat)+1
            if (not r['completed'] or not r['ledger_passed'] or ledger['hands'] != r['hands']
                    or rank != r['rank'] or r['net_points'] != r['final_scores'][seat]-25000
                    or r['wins'] != sum(e['type'] == 'hora' and e.get('actor') == seat for e in log)
                    or r['dealin_hands'] != hand_dealin_count(log, seat)):
                raise ValueError('Cached full-match rank/win/dealin/financial accounting differs')
            for operation, count in r['choices'].items():
                if count != sum(e['type'] == operation and e.get('actor') == seat for e in log): raise ValueError('Actual operation count differs')
            checked += 1
        values = dict(matches=64, seed_groups=8, net_points=statistics.mean(r['net_points'] for r in games),
            hand_win_rate=statistics.mean(r['wins']/r['hands'] for r in games),
            dealin_hand_rate=statistics.mean(r['dealin_hands']/r['hands'] for r in games),
            first_rate=statistics.mean(int(r['rank'] == 1) for r in games), top2_rate=statistics.mean(int(r['rank'] <= 2) for r in games),
            fourth_rate=statistics.mean(int(r['rank'] == 4) for r in games))
        stored = result[name+'_summary']
        if any(abs(values[k]-stored[k]) > 1e-12 for k in values): raise ValueError('Stored aggregate differs from independently counted outcomes')
        summaries[name] = values
    old = {key(r): r for r in original}; comparisons = {}
    for field, fn in [('net_points', lambda r: r['net_points']), ('top2', lambda r: float(r['rank'] <= 2)),
        ('first', lambda r: float(r['rank'] == 1)), ('fourth', lambda r: float(r['rank'] == 4)),
        ('hand_win', lambda r: r['wins']/r['hands']), ('dealin', lambda r: r['dealin_hands']/r['hands'])]:
        clusters = [statistics.mean(fn(r)-fn(old[key(r)]) for r in proposed if r['seed'] == seed) for seed in sorted({r['seed'] for r in proposed})]
        mean = statistics.mean(clusters); radius = 2.3646242510102993*statistics.stdev(clusters)/math.sqrt(8)
        expected = result['comparisons_against_original_route'][field]
        if abs(expected['gain']-mean) > 1e-10 or any(abs(a-b) > 1e-10 for a, b in zip(expected['ci95'], [mean-radius, mean+radius])):
            raise ValueError('Whole-seed paired interval differs')
        comparisons[field] = dict(gain=mean, ci95=[mean-radius, mean+radius], seed_groups=8)
    if any(sha(p) != h for p, h in {**files, **active}.items()) or bundle(identity['directory']) != identity:
        raise ValueError('Comparison/parameter/active-fit evidence changed')
    out.mkdir(parents=True, exist_ok=False)
    write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active))
    summary = dict(status='same_calls388_complete_legal64_cached_independent_accounting_audit_passed',
        checked_complete_game_records=checked, actual_old_route_reproductions_preserved=16, actual_candidate_prefix_games_preserved=2,
        parameter_identity=identity['identity_sha256'], complete_legal_policy_identity=policy_id,
        original_policy_identity=result['original_policy_identity'], summaries=summaries, comparisons=comparisons,
        source_grade_preserved=reg['parent'], shadow_counter_scope_not_extended='new62_games_only',
        new_inference_games_labels_or_optimizer_updates=0, full_v8_goal_not_proven=True,
        fit_or_release_admitted=False, final_grid_unopened=True, active_fit_dependencies_unchanged=len(active),
        raw_records_private=True, node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
    write(out/'summary-private.json', summary); print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'out', 'terminal', 'exit'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); audit(a.source.resolve(), a.out.resolve(), a.terminal.resolve(), a.exit.resolve())
