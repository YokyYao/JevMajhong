"""Independent native teacher coverage, serving eligibility and reach-followup audit.

Read-only companion to the frozen collectors. Never rewrites source rows, targets,
collection summaries or serving policy. Run only inside a bounded GPU allocation.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.collect_curriculum_endgame_cases import sha


def own_discard_number(native_state, public_state):
    """Count physical native discards, cross-checking public tile tokens, not chars."""
    seat = public_state['seat']
    native_count = len(native_state['discards'][seat])
    if native_count != len(public_state['rivers'][seat].split()):
        raise ValueError('Public river tile count differs from the native observation')
    return native_count + 1


def target_diagnostic(obs, actions, features, request, label, guarded_indices, heuristic_index):
    keys = list(request['questions']['action']['criteria'])
    target_index = keys.index(label)
    eligible = set(guarded_indices(obs, actions, features)) | {i for i, f in enumerate(features) if not f}
    reason = 'allowed'
    reference = heuristic_index(obs, actions, features)
    target, ref = features[target_index], features[reference]
    threats = set(request['state']['riichi']) - {request['state']['seat']}
    fold = bool(threats) and bool(any(features)) and min(f['sh'] for f in features if f) >= 1
    if target_index not in eligible:
        if not target or not ref:
            raise ValueError('A special action was excluded by the serving union')
        if target['sh'] > ref['sh']:
            reason = 'blocked_shanten_increase'
        elif fold and len(threats - set(target['safe'])) > len(threats - set(ref['safe'])):
            reason = 'blocked_rule_risk'
        else:
            raise ValueError('Unexplained native serving exclusion')
    return dict(reason=reason, teacher_target_allowed=target_index in eligible,
                teacher_kind=request['questions']['action']['criteria'][label]['type'],
                opponent_riichi=len(threats), teacher_shanten=target.get('sh'),
                reference_shanten=ref.get('sh'), fold_rule_active=fold,
                eligible_keys=[keys[i] for i in sorted(eligible)])


def audit(out):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS') or socket.gethostname().startswith(('lo-', 'login')):
        raise RuntimeError('A scheduler GPU allocation on a compute node is required')
    out.mkdir(parents=True, exist_ok=False)
    read = lambda p: json.loads(p.read_text())
    reg_path = ROOT/'runs/curriculum-optimization-v2/fit-plan/registered.json'
    reg = read(reg_path)
    sys.path.insert(0, reg['source'])
    sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from riichienv import RiichiEnv, GameRule
    from jevmahjong.core.state import build_request, legal_candidates
    from jevmahjong.evaluation.agents import guarded_indices, heuristic_index
    from jevmahjong.policy.unified import serving_request
    from tools.audit_curriculum_basic_teacher import visible
    from tools.curriculum_risk_teacher import guard_diagnostic
    cohorts = {
        'basic': ('curriculum-basic-teacher-data-v1', 'basic_teacher_native_normalized_encoder_and_available_ancestral_audit_passed'),
        'advanced': ('curriculum-advanced-teacher-data-v1', 'advanced_teacher_native_normalized_encoder_basic_and_ancestral_audit_passed'),
        'risk': ('curriculum-risk-teacher-data-v1', 'risk_teacher_native_normalized_encoder_replay_and_guard_audit_passed'),
    }
    files = {str(p.resolve()): sha(p) for p in (
        Path(__file__), reg_path, Path(reg['source'])/'jevmahjong/core/state.py',
        Path(reg['source'])/'jevmahjong/evaluation/agents.py', ROOT/'tools/curriculum_risk_teacher.py',
        ROOT/'tests/core/test_curriculum_teacher_native_coverage_v1.py',
        ROOT/'docs/experiments/plans/curriculum-teacher-native-coverage-20261123.md')}
    raw_path = ROOT/'runs/curriculum-basic-teacher-data-v1/data/source-matches-private.jsonl'
    files[str(raw_path)] = sha(raw_path)
    starts = {}
    for line in raw_path.open():
        match = json.loads(line)
        for index, event in enumerate(e for e in match['mjai'] if e['type'] == 'start_kyoku'):
            starts[(match['source']['seed'], index)] = event
    totals, followups, corrected, bad_reconstructed = {}, {}, {}, {}
    seen = set()
    with (out/'case-diagnostics-private.jsonl').open('x') as details:
        for cohort, (directory, required_status) in cohorts.items():
            audit_root = ROOT/'runs'/directory/'audit-v1'
            summary_path = audit_root/'summary-private.json'
            summary = read(summary_path)
            if summary['status'] != required_status:
                raise ValueError('Completed independent source audit required')
            files[str(summary_path)] = sha(summary_path)
            collected_path = audit_root.parent/'data/summary-private.json'
            collected = read(collected_path)
            files[str(collected_path)] = sha(collected_path)
            for split, spec in summary['files'].items():
                path = ROOT/spec['path']
                if sha(path) != spec['sha256']:
                    raise ValueError('Audited teacher data changed')
                files[str(path)] = spec['sha256']
                c, after, histogram, wrong = Counter(), Counter(), Counter(), Counter()
                for line in path.open():
                    row = json.loads(line)
                    meta = row['_meta']
                    identity = meta['id']
                    if identity in seen or meta['split'] != split:
                        raise ValueError('Duplicate decision or changed split')
                    seen.add(identity)
                    start = starts[(meta['source_logical_seed'], meta['source_hand_index'])]
                    state = row['state']
                    if start['scores'] != meta['source_initial_current_hand_scores']:
                        raise ValueError('Source starting scores changed')
                    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(),
                                    seed=meta['source_engine_seed'] + meta['source_hand_index'])
                    observations = env.reset(oya=state['dealer'], round_wind=state['round_wind'],
                        honba=state['honba'], scores=start['scores'], kyotaku=start['kyotaku'])
                    action_id = lambda a: (int(a.action_type), a.tile, tuple(sorted(a.consume_tiles)))
                    for batch in meta['source_trace']:
                        if {a['player'] for a in batch} != set(observations):
                            raise ValueError('Source trace omits an active actor')
                        selected = {}
                        for saved in batch:
                            menu = {action_id(a): a for a in legal_candidates(observations[saved['player']])}
                            selected[saved['player']] = menu[(saved['type'], saved['tile'], tuple(sorted(saved['consumed'])))]
                        observations = env.step(selected)
                    obs = observations[meta['seat']]
                    rebuilt, actions, features = build_request(obs)
                    if serving_request(rebuilt) != visible(row):
                        raise ValueError('Full actual native/normalized serving request mismatch')
                    native = obs.to_dict()
                    turn = own_discard_number(native, rebuilt['state'])
                    histogram['own_discard_number_' + str(turn)] += 1
                    wrong['own_discard_number_' + str(len(rebuilt['state']['rivers'][meta['seat']]) + 1)] += 1
                    q = row['questions']['action']
                    result = target_diagnostic(obs, actions, features, rebuilt, q['label'], guarded_indices, heuristic_index)
                    c['rows'] += 1
                    c['guard:' + result['reason']] += 1
                    c['stratum:' + meta['curriculum_stratum']] += 1
                    c['selected:' + result['teacher_kind']] += 1
                    c['threats:' + str(result['opponent_riichi']) + ':' + result['reason']] += 1
                    if result['teacher_shanten'] == 0:
                        c['teacher_tenpai:' + result['reason']] += 1
                    if cohort == 'risk':
                        independent = guard_diagnostic(obs, actions, features, rebuilt, q['label'], guarded_indices, heuristic_index)
                        if independent != meta['guard_diagnostic'] or independent['reason'] != result['reason']:
                            raise ValueError('Risk guard diagnostic changed')
                    if result['teacher_kind'] == 'reach':
                        if set(observations) != {meta['seat']}:
                            raise ValueError('Unexpected simultaneous actors at a reach declaration')
                        index = list(rebuilt['questions']['action']['criteria']).index(q['label'])
                        next_observations = env.step({meta['seat']: actions[index]})
                        after['source_selected_reach_executed'] += 1
                        if meta['seat'] in next_observations:
                            next_obs = next_observations[meta['seat']]
                            kinds = sorted({json.loads(a.to_mjai())['type'] for a in legal_candidates(next_obs)})
                            after['same_seat_next_menu:' + ','.join(kinds)] += 1
                            after['same_seat_next_actions_' + str(len(legal_candidates(next_obs)))] += 1
                        else:
                            after['next_actors_do_not_include_declaring_seat'] += 1
                    details.write(json.dumps(dict(cohort=cohort, split=split, id=identity,
                        native_own_discard_number=turn, **result), separators=(',', ':')) + '\n')
                    if c['rows'] % 2000 == 0:
                        print(json.dumps(dict(stage='teacher_native_coverage', cohort=cohort, split=split, rows=c['rows'])), flush=True)
                if c['rows'] != spec['records']:
                    raise ValueError('Audited cohort not completely reconstructed')
                key = cohort + ':' + split
                totals[key], followups[key], corrected[key], bad_reconstructed[key] = dict(c), dict(after), dict(histogram), dict(wrong)
                if cohort == 'risk':
                    recorded = {k: v for k, v in collected['counts'][split].items() if k.startswith('own_discard_number_')}
                    if recorded != dict(wrong):
                        raise ValueError('Reported bad counter cannot be reproduced from river text lengths')
                    for reason in ('allowed', 'blocked_shanten_increase', 'blocked_rule_risk'):
                        if c.get('guard:' + reason, 0) != collected['counts'][split].get('guard:' + reason, 0):
                            raise ValueError('Collector guard counts changed')
    if not all(sha(p) == digest for p, digest in files.items()):
        raise ValueError('A bound input changed during the independent audit')
    result = dict(status='all_teacher_native_turn_guard_and_reach_followups_audited',
        checked_rows=len(seen), counts=totals, native_discard_number_histograms=corrected,
        source_selected_reach_native_followups=followups,
        superseded_risk_text_length_counter_reproduced=True,
        superseded_field='risk collection counts.*.own_discard_number_*',
        authoritative_replacement='native_discard_number_histograms.risk:train and risk:validation',
        source_inputs_labels_splits_and_current_serving_unchanged=True,
        no_model_loaded_or_updated=True, no_new_teacher_labels=True,
        reach_followups_are_interface_checks_not_strategy_or_strength=True,
        private_case_details=True, student_fit_admitted=False)
    for name, value in (
        ('binding-private.json', dict(files=files, node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])),
        ('summary-private.json', result)):
        with (out/name).open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    audit(args.out.resolve())
