"""Native/encoder/one-forward verification for the future rank-win contract.

This uses recorded states and tiny real inference probes, not a new policy
strength benchmark. No candidate-label selection or student update occurs.
"""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import socket
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha
from tools.curriculum_rank_win_input_v2 import (
    serving_request, training_record, continuation_keys, RankWinPolicyAgent,
    RankWinPolicyClient, SCHEMA, METHOD, INSTRUCTION)

PUBLIC_FILES = (
    'tools/curriculum_rank_win_input_v2.py', 'tools/curriculum_rank_win_input_v1.py', 'tools/verify_curriculum_rank_win_input_v2.py',
    'tests/core/test_curriculum_rank_win_input_v1.py',
    'cluster/curriculum-rank-win-input-v2.slurm',
    'docs/experiments/plans/curriculum-native-public-event-20261123.md',
)


class Selector:
    def __init__(self, key): self.key = key; self.calls = 0; self.last = None
    def system_one(self, **request):
        self.calls += 1; self.last = request
        keys = list(request['questions']['action']['criteria'])
        if self.key not in keys: raise ValueError('Expected native selector action absent')
        return dict(answers=dict(action=dict(probabilities={k: float(k == self.key) for k in keys})))


def verify():
    if (not os.environ.get('SLURM_JOB_ID') or not (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
        or not socket.gethostname().startswith('ws-')):raise RuntimeError('Actual bounded GPU allocation required')
    registration = ROOT/'runs/curriculum-progression-v1/fit-plan/registered.json'; reg = read(registration)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(1)
    from types import SimpleNamespace
    from riichienv import ActionType, RiichiEnv, GameRule
    from jevmahjong.policy.unified import serving_request as ordinary, UnifiedPolicyClient
    from jevmahjong.policy import unified as actual_runtime
    from jevmahjong.policy.routing import load_model
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.core.state import build_request, signature
    from jevany.api import SystemOneRequest, to_record
    from jevany.data import api_request
    from jevany.model import DecisionModel, load_preprocessor
    from tools.audit_curriculum_reused_terminal_v1 import restore
    from tools.curriculum_win_context import build_win_request
    manifest_path = ROOT/'runs/curriculum-rank-win-labels-v1/registered-private.json'; manifest = read(manifest_path)
    files = {str((ROOT/name).resolve()): sha(ROOT/name) for name in PUBLIC_FILES}
    for p in (registration, manifest_path): files[str(p.resolve())] = sha(p)
    if (not Path(actual_runtime.__file__).resolve().is_relative_to(Path(reg['source']).resolve())
            or package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']
            or manifest['continuation_model']['identity_sha256'] != reg['curriculum']['reference_identity']):
        raise ValueError('Pinned runtime or frozen continuation changed')
    active = read(ROOT/'runs/curriculum-progression-v1/launch-binding-private.json')['files']
    if any(sha(p) != h for p, h in active.items()): raise ValueError('Active3A changed')
    out = ROOT/'runs/curriculum-rank-win-input-v1/verification-v2'; out.mkdir(parents=True, exist_ok=False)
    wanted = {s['id']: s for s in manifest['selected']}; rows = {}
    for spec in manifest['source_files'].values():
        path = ROOT/spec['path']; files[str(path.resolve())] = sha(path)
        if sha(path) != spec['sha256']: raise ValueError('Original terminal corpus changed')
        for line in path.open():
            row = json.loads(line)
            if row['_meta']['id'] in wanted: rows[row['_meta']['id']] = row
    if set(rows) != set(wanted): raise ValueError('All263 fixed view inputs required')
    tokenizer = load_preprocessor(reg['training_config']['base_load_path'], backbone_adapter='text')
    recipe = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    proxy = SimpleNamespace(decision_mode='pointer', option_isolation=False)
    counts = Counter(); maximum = 0; probes = {}
    for rid, row in rows.items():
        original = copy.deepcopy(row); env, obs = restore(row)
        request, actions, features = build_win_request(obs, env.mjai_log[-1])
        public = serving_request(request); record = training_record(row)
        if serving_request(record) != public or row != original:
            raise ValueError('Train/serve visible projection differs or raw source mutated')
        saved = to_record(SystemOneRequest.model_validate(api_request(record)))
        encoded = DecisionModel.encode(proxy, tokenizer, saved[0], max_state=recipe['max_state'],
            max_branch=recipe['max_branch'], strict=True)
        if (encoded['state_truncated'] or len(encoded['ids']) > recipe['max_packed']
                or saved[1][0]['keys'] != list(request['questions']['action']['criteria'])):
            raise ValueError('Exact future input truncates or native menu reorders')
        maximum = max(maximum, len(encoded['ids']))
        keys = list(request['questions']['action']['criteria'])
        win = next(k for k in keys if request['questions']['action']['criteria'][k]['type'] == 'hora')
        defer = continuation_keys(obs, request, actions, features)
        for chosen in (win, defer[0]):
            model = Selector(chosen); client = RankWinPolicyClient(model); agent = RankWinPolicyAgent(client)
            action = agent.act(obs,current_public_event=env.mjai_log[-1])
            if (model.calls != 1 or model.last != public or signature(action) != signature(actions[keys.index(chosen)])
                    or agent.diagnostics['late_win_model_calls'] != 1):
                raise ValueError('One-model exact native win/continue execution failed')
            counts['scripted_native_one_call_windows'] += 1
        counts['native_views_checked'] += 1
        counts['native_views_with_empty_observation_events'] += not bool(obs.events)
        info = next(iter(request['state']['win_context']['candidates'].values()))
        counts['native_'+info['kind']+'_views'] += 1
        bucket = (info['kind'], wanted[rid]['public_rank'] >= 3)
        probes.setdefault(bucket, request)
    if set(probes) != {(kind, behind) for kind in ('ron', 'tsumo') for behind in (False, True)}:
        raise ValueError('Actual ron/tsumo and leading/behind native coverage required')
    # Ordinary validation must remain byte-equivalent at the actual model API.
    data = read(ROOT/'runs/curriculum-yaku-scenarios-v1/data-v2/data-ready-private.json')
    path = ROOT/data['files']['validation']['path']; files[str(path.resolve())] = sha(path)
    if sha(path) != data['files']['validation']['sha256']: raise ValueError('Original24516 replay changed')
    plain_probes = {}
    for line in path.open():
        row = json.loads(line)
        if serving_request(row) != ordinary(row): raise ValueError('Ordinary replay projection changed')
        counts['ordinary_validation_inputs_identical'] += 1
        kinds = tuple(sorted({v['type'] for v in row['questions']['action']['criteria'].values()}))
        plain_probes.setdefault(kinds, row)
    # Confirm the existing early legal-win shortcut with an actual native case.
    source_path = ROOT/'runs/curriculum-yaku-scenarios-v1/scenarios-v1/validation-unlabelled-scenarios-private.jsonl'
    files[str(source_path.resolve())] = sha(source_path); early = False
    for line in source_path.open():
        source = json.loads(line)['source']; env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=source['native_seed'])
        obs = env.reset(oya=0, wall=source['wall'])[0]
        if any(a.action_type in (ActionType.RON, ActionType.TSUMO) for a in obs.legal_actions()):
            class Never:
                def system_one(self, **kw): raise ValueError('Early win consulted a model')
            chosen = RankWinPolicyAgent(Never()).act(obs)
            if chosen.action_type not in (ActionType.RON, ActionType.TSUMO): raise ValueError('Early win declined')
            early = True; break
    if not early or counts['ordinary_validation_inputs_identical'] != 24516:
        raise ValueError('Full ordinary/early shortcut coverage required')
    # Four conditional calls and two unchanged ordinary calls; no new game benchmark.
    checkpoint = Path(manifest['continuation_model']['directory']); identity = bundle(checkpoint)
    model = load_model(checkpoint); client = RankWinPolicyClient(model)
    from jevany.api import validate_distribution
    for request in probes.values():
        result = client.system_one(**request)
        validate_distribution(result['answers']['action']['probabilities'],
            list(request['questions']['action']['criteria']))
        counts['actual_new_schema_forward_calls'] += 1
    reference = UnifiedPolicyClient(model)
    ordinary_error = 0.
    for request in list(plain_probes.values())[:2]:
        a = reference.system_one(**api_request(request)); b = client.system_one(**api_request(request))
        left = a['answers']['action']['probabilities']; right = b['answers']['action']['probabilities']
        keys = list(request['questions']['action']['criteria'])
        validate_distribution(left, keys); validate_distribution(right, keys)
        ordinary_error = max(ordinary_error, max(abs(left[k]-right[k]) for k in keys))
        if ordinary_error >= 5e-5: raise ValueError('Same-model ordinary inference changed')
        counts['actual_ordinary_same_model_equal'] += 1
    if bundle(checkpoint) != identity or any(sha(p) != h for p, h in {**files, **active}.items()):
        raise ValueError('Verification changed weights or frozen inputs')
    result = dict(status='rank_win_native_input_encoder_and_single_model_route_verified_not_student_fit',
        input_schema=SCHEMA, policy_method=METHOD, counts=dict(counts), max_packed_tokens=maximum,
        actual_ordinary_forward_max_probability_error=ordinary_error,
        early_actual_legal_win_without_model_preserved=True, same_model_and_parameter_bundle=True,
        no_external_residual_network=True, no_optimizer_updates=True,
        explicit_current_public_event_required_by_native_game_loop=True,
        previous_failed_job=318842,missing_event_not_guessed_from_rivers=True,
        tiny_frozen_v8_forward_probes_not_policy_strength=True,
        tsumo_continue_uses_current_same_model_best_guarded_discard=True,
        teacher_uses_frozen_v8_discard_and_future_closed_loop_evaluation_required=True,
        conditional_labels_input_conversion_and_non_win_targets_pending=True,
        production_v8_main_Windows_unchanged=True, final_grid_unused=True,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
    write(out/'summary-private.json', result); write(out/'summary-public.json', result)
    write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active))
    print(json.dumps(result), flush=True)


if __name__ == '__main__': verify()

