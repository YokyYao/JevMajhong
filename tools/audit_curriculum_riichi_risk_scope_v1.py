"""Native teacher-prefix audit of a prospective defensive eligibility scope.

Synthetic preferences test execution and availability only; no student weights,
new labels, new games, training admission or native-strength claim is made.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_riichi_risk_scope_policy_v1 import METHOD, eligible, RiichiRiskScopePolicyAgent
from tools.curriculum_riichi_class_policy_v1 import select

PUBLIC_FILES = (
    'tools/curriculum_riichi_risk_scope_policy_v1.py',
    'tools/audit_curriculum_riichi_risk_scope_v1.py',
    'tests/core/test_curriculum_riichi_risk_scope_policy_v1.py',
    'docs/experiments/plans/curriculum-defensive-scope-20261123.md',
    'cluster/curriculum-defensive-scope-v1.slurm',
)
COHORTS = {
    'basic': ('curriculum-basic-teacher-data-v1', 'basic_teacher_native_normalized_encoder_and_available_ancestral_audit_passed'),
    'advanced': ('curriculum-advanced-teacher-data-v1', 'advanced_teacher_native_normalized_encoder_basic_and_ancestral_audit_passed'),
    'risk': ('curriculum-risk-teacher-data-v1', 'risk_teacher_native_normalized_encoder_replay_and_guard_audit_passed'),
    'post_reach': ('curriculum-post-reach-teacher-data-v2', 'post_reach_teacher_native_normalized_encoder_replay_and_guard_audit_passed'),
}


def audit(out):
    gpu_only(); out.mkdir(parents=True, exist_ok=False)
    reg_path = ROOT/'runs/curriculum-calls-classroute-v1/fit-plan/registered.json'; reg = read(reg_path)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from riichienv import RiichiEnv, GameRule, ActionType
    from jevmahjong.core.state import build_request, legal_candidates
    from jevmahjong.evaluation.agents import guarded_indices, heuristic_index
    from jevmahjong.policy.unified import serving_request
    from jevmahjong.training.base_lora import package_digest
    from jevany.api import validate_distribution
    from tools.audit_curriculum_teacher_native_coverage_v1 import target_diagnostic
    from tools.audit_curriculum_basic_teacher import visible
    if package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']:
        raise ValueError('Pinned native/training runtime changed')
    files = {str((ROOT/p).resolve()): sha(ROOT/p) for p in PUBLIC_FILES}
    active = ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json'
    files.update(read(active)['files']); files[str(active)] = sha(active)
    files[str(reg_path)] = sha(reg_path)
    helper = ROOT/'tools/audit_curriculum_teacher_native_coverage_v1.py'; files[str(helper)] = sha(helper)
    raw_path = ROOT/'runs/curriculum-basic-teacher-data-v1/data/source-matches-private.jsonl'
    files[str(raw_path)] = sha(raw_path); starts = {}
    for line in raw_path.open():
        match = json.loads(line)
        for index, event in enumerate(e for e in match['mjai'] if e['type'] == 'start_kyoku'):
            starts[match['source']['seed'], index] = event
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Immutable active/source input changed')
    signature = lambda a: (int(a.action_type), a.tile, tuple(sorted(a.consume_tiles)))
    totals = {}; seen = set(); unchanged = 0; model_calls = 0
    for cohort, (directory, required_status) in COHORTS.items():
        source = ROOT/'runs'/directory/'audit-v1'; manifest_path = source/'summary-private.json'
        manifest = read(manifest_path)
        if manifest['status'] != required_status: raise ValueError('Complete native-source audit required')
        files[str(manifest_path)] = sha(manifest_path)
        binding = source/'audit-binding-private.json'
        for p, h in read(binding)['files'].items():
            if p in files and files[p] != h: raise ValueError('Conflicting source identity')
            files[p] = h
        files[str(binding)] = sha(binding)
        for split, spec in manifest['files'].items():
            path = ROOT/spec['path']
            if sha(path) != spec['sha256']: raise ValueError('Audited source dataset changed')
            files[str(path)] = spec['sha256']; counted = Counter()
            for line in path.open():
                row = json.loads(line); meta = row['_meta']; state = row['state']
                if meta['id'] in seen or meta['split'] != split: raise ValueError('Duplicate or mis-split source decision')
                seen.add(meta['id']); start = starts[meta['source_logical_seed'], meta['source_hand_index']]
                if start['scores'] != meta['source_initial_current_hand_scores']:
                    raise ValueError('Source starting scores changed')
                env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(),
                                seed=meta['source_engine_seed']+meta['source_hand_index'])
                observations = env.reset(oya=state['dealer'], round_wind=state['round_wind'],
                    honba=state['honba'], scores=start['scores'], kyotaku=start['kyotaku'])
                for batch in meta['source_trace']:
                    if {a['player'] for a in batch} != set(observations): raise ValueError('Incomplete physical source batch')
                    selected = {}
                    for saved in batch:
                        menu = {signature(a): a for a in legal_candidates(observations[saved['player']])}
                        selected[saved['player']] = menu[saved['type'], saved['tile'], tuple(sorted(saved['consumed']))]
                    observations = env.step(selected)
                obs = observations[meta['seat']]; rebuilt, actions, features = build_request(obs)
                if serving_request(rebuilt) != visible(row): raise ValueError('Exact native public input/menu mismatch')
                q = rebuilt['questions']['action']; keys = list(q['criteria']); types = [q['criteria'][k]['type'] for k in keys]
                old = sorted(set(guarded_indices(obs, actions, features)) | {i for i, f in enumerate(features) if not f})
                allowed, scope = eligible(types, rebuilt['state'], old)
                target = keys.index(row['questions']['action']['label'])
                diagnostic = target_diagnostic(obs, actions, features, rebuilt,
                    row['questions']['action']['label'], guarded_indices, heuristic_index)
                counted['rows'] += 1; counted['scope' if scope else 'outside_scope'] += 1
                counted['old_guard:'+diagnostic['reason']] += 1
                counted['teacher_target_now_available'] += target in allowed
                counted['newly_available_teacher_targets'] += target not in old and target in allowed
                counted['scope_open_hand' if rebuilt['state']['melds'][meta['seat']] else 'scope_closed_hand'] += scope
                counted['scope_threats_'+str(len(set(rebuilt['state']['riichi'])-{meta['seat']}))] += scope
                if cohort == 'risk' and (not scope or target not in allowed):
                    raise ValueError('Actual closed-risk teacher menu is outside prospective defensive support')
                if not scope and allowed != old: raise ValueError('Outside-scope candidate eligibility changed')
                # Synthetic teacher-preferred scores, not a teacher inference
                # or a claim that these moves improve the student's outcome.
                p = [(.95 if i == target else .05/(len(keys)-1)) for i in range(len(keys))] if len(keys) > 1 else [1.]
                expected, _ = select(types, p, allowed); baseline, _ = select(types, p, old)
                if scope and 'reach' in types:
                    reach = types.index('reach'); discards = [i for i, kind in enumerate(types) if kind == 'dahai']
                    for probability in (.4, .5, .6):
                        boundary = [(1-probability)/len(discards) if i in discards else probability for i in range(len(types))]
                        previous, _ = select(types, boundary, old); current, _ = select(types, boundary, allowed)
                        if (types[previous] == 'reach') != (types[current] == 'reach'):
                            raise ValueError('The frozen RI class operating point changed')
                        counted['RI_boundary_class_comparisons'] += 1
                if not scope:
                    if expected != baseline: raise ValueError('Outside-scope RI/call/kan decision changed')
                    unchanged += 1
                if scope and expected != target: raise ValueError('Legal teacher-preferred defensive action remains blocked')
                class Client:
                    def __init__(self): self.calls = 0
                    def system_one(self, **request):
                        if request != rebuilt: raise ValueError('Agent changed exact native request')
                        self.calls += 1
                        return dict(answers=dict(action=dict(probabilities=dict(zip(keys, p)))))
                client = Client(); agent = RiichiRiskScopePolicyAgent(client, validator=validate_distribution)
                actual = agent.act(obs)
                forced = len(actions) == 1 or any(a.action_type in (ActionType.RON, ActionType.TSUMO) for a in actions)
                if forced:
                    if client.calls: raise ValueError('Shortcut invoked a model')
                    counted['forced_source_windows'] += 1
                elif signature(actual) != signature(actions[expected]) or client.calls != 1:
                    raise ValueError('Actual native prospective agent choice/forward count differs')
                model_calls += client.calls
                if counted['rows'] % 2000 == 0:
                    print(json.dumps(dict(cohort=cohort, split=split, checked=counted['rows'])), flush=True)
            if counted['rows'] != spec['records']: raise ValueError('Incomplete native source coverage')
            totals[cohort+':'+split] = dict(counted)
    if len(seen) != 45351: raise ValueError('Complete four-cohort native teacher coverage required')
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Bound source or active fit changed')
    result = dict(status='prospective_defensive_scope_native_source_availability_audited_not_fit_admitted',
        method=METHOD, checked_native_source_windows=len(seen), counts=totals,
        outside_scope_synthetic_choices_unchanged=unchanged, synthetic_agent_forward_calls=model_calls,
        actual_model_parameters_loaded=False, synthetic_preferences_not_policy_or_optimality_labels=True,
        source_actions_inputs_splits_unchanged=True, new_games_or_labels=0, student_optimizer_updates=0,
        active_calls_fit_and_serving_contract_unchanged=True, risk_fit_admitted=False, automatic_release=False,
        actual_future_parent_native_strength_comparison_still_required=True, raw_private=True)
    write(out/'summary-private.json', result)
    write(out/'binding-private.json', dict(files=files))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out.resolve())
