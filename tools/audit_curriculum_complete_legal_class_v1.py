"""Immutable teacher-menu and native execution preflight, never a model score."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_complete_legal_class_policy_v1 import METHOD, select, CompleteLegalClassPolicyAgent
from tools.curriculum_riichi_class_policy_v1 import select as old_select
from tools.audit_curriculum_riichi_risk_scope_v1 import COHORTS

PUBLIC_FILES = (
    'tools/curriculum_complete_legal_class_policy_v1.py',
    'tools/audit_curriculum_complete_legal_class_v1.py',
    'tests/core/test_curriculum_complete_legal_class_policy_v1.py',
    'configs/diagnostics/curriculum-complete-legal-class-v1.json',
    'docs/experiments/plans/curriculum-complete-legal-class-20261123.md',
    'cluster/curriculum-complete-legal-class-v1.slurm',
)


def preference(size, target):
    return [1.] if size == 1 else [.95 if i == target else .05/(size-1) for i in range(size)]


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
    from tools.audit_curriculum_basic_teacher import visible
    from tools.audit_curriculum_teacher_native_coverage_v1 import target_diagnostic
    from tools.curriculum_observer_decline_fixtures_v3 import SEED, physical_wall, discard, pass_responses, fixture
    from tools.curriculum_endgame_ledger import ledger
    files = {str((ROOT/p).resolve()): sha(ROOT/p) for p in PUBLIC_FILES}
    files[str(reg_path)] = sha(reg_path)
    active_path = ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json'
    active = read(active_path)['files']; files[str(active_path)] = sha(active_path)
    for path in ('tools/curriculum_riichi_class_policy_v1.py', 'tools/audit_curriculum_riichi_risk_scope_v1.py',
                 'tools/audit_curriculum_basic_teacher.py', 'tools/audit_curriculum_teacher_native_coverage_v1.py',
                 'tools/curriculum_observer_decline_fixtures_v3.py', 'tools/curriculum_endgame_ledger.py'):
        files[str(ROOT/path)] = sha(ROOT/path)
    if package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']:
        raise ValueError('Pinned runtime changed')
    config = read(ROOT/PUBLIC_FILES[3])
    if config['method'] != METHOD or config['expected_teacher_rows'] != 82419:
        raise ValueError('Registered preflight contract changed')
    sources = dict(COHORTS, ordinary_completion=('curriculum-ordinary-completion-v1',
        'ordinary_completion_native_event_label_encoder_split_audit_passed_not_fit_admitted'))
    seen = set(); selected = {}; totals = {}; ordinary_excluded = Counter()
    for cohort, (directory, expected_status) in sources.items():
        root = ROOT/'runs'/directory/'audit-v1'; manifest_path = root/'summary-private.json'; manifest = read(manifest_path)
        if manifest['status'] != expected_status: raise ValueError('Completed independent source audit required')
        files[str(manifest_path)] = sha(manifest_path)
        binding = root/('completion-binding-private.json' if cohort == 'ordinary_completion' else 'audit-binding-private.json')
        files[str(binding)] = sha(binding)
        for path, digest in read(binding)['files'].items():
            if path in files and files[path] != digest: raise ValueError('Conflicting immutable source identity')
            files[path] = digest
        for split, spec in manifest['files'].items():
            path = ROOT/spec['path']; files[str(path)] = spec['sha256']
            if sha(path) != spec['sha256']: raise ValueError('Audited teacher data changed')
            count = Counter()
            for line in path.open():
                row = json.loads(line); meta = row['_meta']; q = row['questions']['action']
                if meta['id'] in seen or meta['split'] != split: raise ValueError('Duplicate decision ID or changed split')
                seen.add(meta['id']); keys = list(q['criteria']); kinds = [q['criteria'][k]['type'] for k in keys]
                target = keys.index(q['label']); probability = preference(len(keys), target)
                if select(kinds, probability)[0] != target: raise ValueError('Legal teacher-preferred menu target is blocked')
                reason = meta.get('guard_diagnostic', {}).get('reason', 'not_saved')
                selected.setdefault((cohort, split, 'target', kinds[target], reason), row)
                for kind in sorted(set(kinds)): selected.setdefault((cohort, split, 'offered', kind), row)
                count['rows'] += 1; count['target:'+kinds[target]] += 1
                if cohort == 'ordinary_completion' and reason != 'allowed': ordinary_excluded[split] += 1
            if count['rows'] != spec.get('records', spec.get('rows')): raise ValueError('Source row count changed')
            totals[cohort+':'+split] = dict(count)
    if len(seen) != 82419 or dict(ordinary_excluded) != config['expected_ordinary_guard_exclusions']:
        raise ValueError('Whole audited teacher coverage or ordinary exclusions differ')
    if any(sha(p) != h for p, h in {**files, **active}.items()): raise ValueError('Bound source or active fit changed')
    # Deduplicate a row selected by multiple registered coverage keys.
    subset = {row['_meta']['id']: row for row in selected.values()}
    write(out/'registered-private.json', dict(files=files, active_fit_files=active, method=METHOD,
        deterministic_selection_keys=[list(k) for k in selected], selected_ids=list(subset),
        synthetic_preferences_not_new_teacher_labels=True))
    raw = ROOT/'runs/curriculum-basic-teacher-data-v1/data/source-matches-private.jsonl'
    files[str(raw)] = sha(raw); starts = {}
    for line in raw.open():
        match = json.loads(line)
        for index, event in enumerate(e for e in match['mjai'] if e['type'] == 'start_kyoku'):
            starts[match['source']['seed'], index] = event
    signature = lambda a: (int(a.action_type), a.tile, tuple(sorted(a.consume_tiles)))
    native = Counter(); offered = Counter(); restored_ordinary = Counter()
    with (out/'native-subset-private.jsonl').open('x') as stream:
        for identity, row in subset.items():
            meta = row['_meta']; state = row['state']; start = starts[meta['source_logical_seed'], meta['source_hand_index']]
            if start['scores'] != meta['source_initial_current_hand_scores']: raise ValueError('Original source scores differ')
            env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=meta['source_engine_seed']+meta['source_hand_index'])
            observations = env.reset(oya=state['dealer'], round_wind=state['round_wind'], honba=state['honba'], scores=start['scores'], kyotaku=start['kyotaku'])
            for batch in meta['source_trace']:
                if {a['player'] for a in batch} != set(observations): raise ValueError('Incomplete physical source batch')
                actions = {}
                for saved in batch:
                    menu = {signature(a): a for a in legal_candidates(observations[saved['player']])}
                    actions[saved['player']] = menu[saved['type'], saved['tile'], tuple(sorted(saved['consumed']))]
                observations = env.step(actions)
                if env.done(): raise ValueError('Physical source trace ended before target')
            obs = observations[meta['seat']]; request, actions, features = build_request(obs)
            if serving_request(request) != visible(row): raise ValueError('Exact native public input/menu differs')
            q = request['questions']['action']; keys = list(q['criteria']); kinds = [q['criteria'][k]['type'] for k in keys]
            if len(actions) < 2 or any(a.action_type in (ActionType.RON, ActionType.TSUMO) for a in actions):
                raise ValueError('A teacher network source unexpectedly contains a forced shortcut')
            old = sorted(set(guarded_indices(obs, actions, features)) | {i for i, f in enumerate(features) if not f})
            diagnostic = target_diagnostic(obs, actions, features, request, row['questions']['action']['label'], guarded_indices, heuristic_index)
            for target in range(len(keys)):
                p = preference(len(keys), target)
                class Client:
                    def __init__(self): self.calls = 0
                    def system_one(self, **actual):
                        if actual != request: raise ValueError('Agent changed the actual native request')
                        self.calls += 1
                        return dict(answers=dict(action=dict(probabilities=dict(zip(keys, p)))))
                client = Client(); agent = CompleteLegalClassPolicyAgent(client, validator=validate_distribution)
                actual = agent.act(obs)
                if client.calls != 1 or signature(actual) != signature(actions[target]):
                    raise ValueError('Actual native agent blocks a legal preferred candidate or invokes extra forwards')
                native['candidate_forward_checks'] += 1; offered[kinds[target]] += 1
                native['previously_excluded_candidates_recovered'] += target not in old
            if set(kinds) == {'reach', 'dahai'}:
                reach = kinds.index('reach'); discards = [i for i, k in enumerate(kinds) if k == 'dahai']
                for p_ri in config['RI_boundary_probabilities']:
                    p = [p_ri if i == reach else (1-p_ri)/len(discards) for i in range(len(keys))]
                    before, _ = old_select(kinds, p, old); after, _ = select(kinds, p)
                    if (kinds[before] == 'reach') != (kinds[after] == 'reach'): raise ValueError('RI threshold changed')
                    client = Client(); actual = CompleteLegalClassPolicyAgent(client, validator=validate_distribution).act(obs)
                    if client.calls != 1 or signature(actual) != signature(actions[after]): raise ValueError('Actual RI boundary differs')
                    native['RI_boundary_actual_forward_checks'] += 1
            native['source_prefixes_checked'] += 1
            if meta.get('curriculum_stratum') == 'ordinary_completion' and diagnostic['reason'] != 'allowed':
                restored_ordinary[meta['split']] += 1
            stream.write(json.dumps(dict(id=identity, native_candidates=len(keys), old_guard=diagnostic))+'\n')
    # Actual native shortcuts, with a client that raises if any inference occurs.
    class NeverClient:
        def system_one(self, **request): raise AssertionError('A legal win invoked the model')
    ron_env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=SEED)
    observations = ron_env.reset(oya=0, wall=physical_wall('temporary_draw'))
    observations = discard(ron_env, observations, 0, 112); observations = pass_responses(ron_env, observations)
    observations = discard(ron_env, observations, 1, 73)
    ron = CompleteLegalClassPolicyAgent(NeverClient()).act(observations[0])
    if ron.action_type != ActionType.RON: raise ValueError('Actual native RON shortcut failed')
    batch = {p: ron if p == 0 else next(a for a in legal_candidates(o) if a.action_type == ActionType.PASS) for p, o in observations.items()}
    ron_env.step(batch); ledger(ron_env.mjai_log, ron_env.scores())
    tsumo_env, tsumo_obs, _ = fixture('temporary_tsumo')
    tsumo = CompleteLegalClassPolicyAgent(NeverClient()).act(tsumo_obs)
    if tsumo.action_type != ActionType.TSUMO: raise ValueError('Actual native TSUMO shortcut failed')
    tsumo_env.step({0: tsumo}); ledger(tsumo_env.mjai_log, tsumo_env.scores())
    if not {'dahai', 'reach', 'none', 'chi', 'pon', 'ankan', 'daiminkan', 'kakan'} <= set(offered):
        raise ValueError('Native candidate coverage missing a required operation')
    if any(sha(p) != h for p, h in {**files, **active}.items()): raise ValueError('Immutable inputs changed during preflight')
    write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active))
    result = dict(status='complete_legal_fixed_RI_menu_and_native_interface_preflight_passed_not_fit_admitted',
        method=METHOD, all_teacher_menus_checked=len(seen), counts=totals,
        ordinary_old_guard_excluded_targets_now_available=dict(ordinary_excluded),
        native_subset=dict(native), native_candidate_kinds=dict(offered),
        actual_RON_and_TSUMO_shortcuts_executed_and_ledgers_checked=2,
        active_fit_dependencies_unchanged=len(active), actual_model_parameters_loaded=False,
        synthetic_preferences_are_interface_probes_not_model_quality=True,
        student_updates=0, new_games_or_teacher_labels=0, actual_parent_native_strength_comparison_pending=True,
        fit_or_production_admitted=False, raw_records_private=True, node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
    write(out/'summary-private.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    audit(p.parse_args().out.resolve())
