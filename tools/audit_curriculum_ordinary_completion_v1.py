"""Independent native/event-label/encoder/split audit of missing ordinary moves."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import socket
import sys
import tomllib
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write


def audit(source, out, terminal, exit_path):
    gpu_only()
    from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
    scheduler_exit(read(exit_path), terminal.read_text(), ROOT/'cluster/curriculum-ordinary-completion-v1.slurm')
    ready = read(source/'summary-private.json'); registration = read(source/'registered-private.json')
    binding = read(source/'completion-binding-private.json')
    if (ready['status'] != 'observed_ordinary_completion_collected_not_fit_admitted'
            or ready['counts']['exact_complete_sources'] != 256
            or not all(sha(p) == h for p, h in binding['files'].items())):
        raise ValueError('Actual complete immutable ordinary collection required')
    original = ROOT/registration['config']['source_root']; parent = read(original/'registered-private.json')
    reg = read(ROOT/parent['config']['source_registration'])
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    from riichienv import RiichiEnv, GameRule, Phase, ActionType
    from jevmahjong.core.state import build_request, legal_candidates
    from jevmahjong.data.base_lora_pool import normalized_request
    from jevmahjong.policy.unified import serving_request
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.agents import guarded_indices, heuristic_index
    from jevany.api import SystemOneRequest, to_record
    from jevany.data import api_request
    from jevany.model import DecisionModel, load_preprocessor
    from tools.audit_curriculum_basic_teacher import visible, fingerprint
    from tools.curriculum_risk_teacher import guard_diagnostic
    from tools.curriculum_win_context import public_tile
    import torch
    torch.set_num_threads(4)
    if (package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']
            or package_digest(Path(reg['training_config']['upstream_source'])/'jevany') != reg['training_config']['upstream_source_sha256']):
        raise ValueError('Pinned source/encoder changed')
    rawpath = original/'source-matches-private.jsonl'; sources = {r['source']['seed']: r for r in (json.loads(line) for line in rawpath.open())}
    starts = {seed: [i for i, e in enumerate(r['mjai']) if e['type'] == 'start_kyoku'] for seed, r in sources.items()}
    files = {str(p.resolve()): sha(p) for p in (Path(__file__), source/'summary-private.json', source/'registered-private.json',
        source/'completion-binding-private.json', rawpath, terminal, exit_path,
        ROOT/'tools/audit_curriculum_basic_teacher.py', ROOT/'tools/curriculum_risk_teacher.py',
        ROOT/'tools/curriculum_win_context.py', ROOT/'cluster/curriculum-ordinary-completion-audit-v1.slurm',
        ROOT/'docs/experiments/plans/curriculum-ordinary-completion-audit-20261123.md')}
    clean_path = ROOT/'runs/curriculum-reproduction-v1/clean-data-v1/data-ready.json'; clean = read(clean_path)
    reserved_path = clean_path.parent/'reserved-public-groups.json'; reserved = set(read(reserved_path))
    files[str(clean_path)] = sha(clean_path); files[str(reserved_path)] = sha(reserved_path)
    refs = []
    for stage in clean['stages'].values(): refs.extend(stage.items())
    for path in (ROOT/'runs/curriculum-calls-post-reach-data-v1/data/data-ready-private.json',
                 ROOT/'runs/curriculum-risk-teacher-data-v1/audit-v1/summary-private.json'):
        files[str(path)] = sha(path); refs.extend(read(path)['files'].items())
    old_inputs = {s: set() for s in ('train', 'validation')}; old_groups = {s: set() for s in old_inputs}
    old_ids = set(); physical_splits = {}; visited = set()
    for split, spec in refs:
        path = ROOT/spec['path']
        if (split, str(path)) in visited: continue
        visited.add((split, str(path))); files[str(path)] = sha(path)
        if files[str(path)] != spec['sha256']: raise ValueError('Earlier admitted data changed')
        for line in path.open():
            row = json.loads(line); m = row['_meta']; old_inputs[split].add(fingerprint(row)); old_groups[split].add(m['group_id']); old_ids.add(m['id'])
            if 'source_logical_seed' in m:
                seed = m['source_logical_seed']
                if seed in physical_splits and physical_splits[seed] != split: raise ValueError('An old physical family crosses splits')
                physical_splits[seed] = split
    if len(reserved) != 675: raise ValueError('Original675 reserved families required')
    recipe = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    tokenizer = load_preprocessor(reg['training_config']['base_load_path'], backbone_adapter='text')
    proxy = SimpleNamespace(decision_mode='pointer', option_isolation=False)
    active = read(ROOT/'runs/curriculum-calls-classroute-v1/fit-launch-binding-private.json')['files']
    out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(files=files,
        direct_label_reference='original source MJAI event offset, independent of collector selected-action receipt',
        no_backbone_or_policy_loaded=True))
    inputs = {s: set() for s in old_inputs}; groups = {s: set() for s in old_inputs}; ids = set()
    counts = {s: Counter() for s in old_inputs}; lengths = {s: Counter() for s in old_inputs}; output_files = {}
    for split, spec in ready['files'].items():
        path = ROOT/spec['path']; files[str(path)] = sha(path)
        if files[str(path)] != spec['sha256']: raise ValueError('Collected ordinary decisions changed')
        dest = out/f'{split}-normalized-private.jsonl'
        with dest.open('x') as stream:
            for line in path.open():
                row = json.loads(line); m = row['_meta']; original_record = sources[m['source_logical_seed']]
                family = original_record['source']; start_index = starts[m['source_logical_seed']][m['source_hand_index']]
                start = original_record['mjai'][start_index]; offset = m['observed_source_event_offset']
                event = original_record['mjai'][offset]; q = row['questions']['action']
                if (m['id'] in ids or m['id'] in old_ids or m['split'] != split or family['split'] != split
                        or family['group_id'] != m['group_id'] or m['group_id'] in reserved
                        or m['source_engine_seed'] != original_record['engine_seed'] or physical_splits[m['source_logical_seed']] != split
                        or m['source_initial_current_hand_scores'] != start['scores']
                        or not start_index < offset < len(original_record['mjai'])
                        or event['type'] != 'dahai' or event['actor'] != m['seat']):
                    raise ValueError('Changed physical family, observed label offset or split')
                env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=m['source_engine_seed']+m['source_hand_index'])
                observations = env.reset(oya=start['oya'], round_wind=int(public_tile(start['bakaze'])[0])-1,
                    honba=start['honba'], scores=start['scores'], kyotaku=start['kyotaku'])
                identity = lambda a: (int(a.action_type), a.tile, tuple(sorted(a.consume_tiles)))
                for batch in m['source_trace']:
                    if {a['player'] for a in batch} != set(observations): raise ValueError('Replay omitted an active actor')
                    selected = {}
                    for saved in batch:
                        menu = {identity(a): a for a in legal_candidates(observations[saved['player']])}
                        wanted = saved['type'], saved['tile'], tuple(sorted(saved['consumed']))
                        if wanted not in menu: raise ValueError('A stored physical action is illegal')
                        selected[saved['player']] = menu[wanted]
                    observations = env.step(selected)
                    if env.done(): raise ValueError('Ordinary trace settled before the target')
                hand_start = next(i for i, e in enumerate(env.mjai_log) if e['type'] == 'start_kyoku')
                if env.mjai_log[hand_start:] != original_record['mjai'][start_index:offset]:
                    raise ValueError('Full physical current-hand prefix differs from original source')
                if env.phase != Phase.WaitAct or set(observations) != {m['seat']}:
                    raise ValueError('Target is not a single-actor WaitAct discard')
                obs = observations[m['seat']]; rebuilt, menu, feats = build_request(obs); native = obs.to_dict()
                if (rebuilt != visible(row) or native['melds'][m['seat']] or any(native['riichi_declared'])
                        or len(menu) < 2 or any(a.action_type != ActionType.DISCARD for a in menu)
                        or env.mjai_log[-1]['type'] != 'tsumo' or env.mjai_log[-1]['actor'] != m['seat']):
                    raise ValueError('Independent public scope/input reconstruction failed')
                matches = [k for k, a in q['criteria'].items() if a['type'] == 'dahai' and public_tile(a['pai']) == public_tile(event['pai'])]
                if len(matches) != 1 or q['label'] != matches[0]: raise ValueError('Label is not the uniquely observed source discard')
                diag = guard_diagnostic(obs, menu, feats, rebuilt, q['label'], guarded_indices, heuristic_index)
                if diag != m['guard_diagnostic']: raise ValueError('Stored execution-eligibility diagnostic differs')
                item = normalized_request(row, m['group_id'], 'expert_replay', m['source_step'])
                if visible(item) != serving_request(rebuilt) or item['questions']['action']['label'] != q['label']:
                    raise ValueError('Normalization changed serving input or the observed label')
                item['_meta'] = copy.deepcopy(m); item['questions']['action']['src'] = 'curriculum3:teacher:ordinary_completion'
                key = fingerprint(item); other = 'validation' if split == 'train' else 'train'
                if (key in inputs[split] or key in old_inputs[split] or key in inputs[other]
                        or key in old_inputs[other] or m['group_id'] in old_groups[other]):
                    raise ValueError('New normalized input duplicates earlier teacher data or opposite split')
                ids.add(m['id']); inputs[split].add(key); groups[split].add(m['group_id'])
                record, mapping = to_record(SystemOneRequest.model_validate(api_request(item)))
                encoded = DecisionModel.encode(proxy, tokenizer, record, max_state=recipe['max_state'], max_branch=recipe['max_branch'], strict=True)
                if (encoded['state_truncated'] or len(encoded['ids']) > recipe['max_packed']
                        or mapping[0]['keys'] != list(q['criteria'])): raise ValueError('Actual tokenizer truncated or reordered this menu')
                cc = counts[split]; cc['rows'] += 1; cc['direct_source_event_labels_checked'] += 1
                cc['guard:'+diag['reason']] += 1; cc['own_discard_number_'+str(len(native['discards'][m['seat']])+1)] += 1
                lengths[split]['max_packed'] = max(lengths[split]['max_packed'], len(encoded['ids']))
                lengths[split]['max_state'] = max(lengths[split]['max_state'], encoded['seg'].count(0))
                stream.write(json.dumps(item)+'\n')
                if len(ids) % 1000 == 0: print(json.dumps(dict(stage='ordinary_completion_input_audit', rows=len(ids))), flush=True)
        output_files[split] = dict(path=str(dest.relative_to(ROOT)), sha256=sha(dest), rows=counts[split]['rows'])
    if len(ids) != ready['counts']['observed_new_discard_labels'] or groups['train'] & groups['validation']:
        raise ValueError('New observation count or source split differs')
    if not all(sha(p) == h for p, h in files.items()) or not all(sha(p) == h for p, h in active.items()):
        raise ValueError('Source or active training dependencies changed during audit')
    write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active))
    result = dict(status='ordinary_completion_native_event_label_encoder_split_audit_passed_not_fit_admitted',
        rows=len(ids), counts={s: dict(v) for s, v in counts.items()}, tokens={s: dict(v) for s, v in lengths.items()},
        files=output_files, groups={s: len(v) for s, v in groups.items()},
        complete_physical_current_hand_prefixes_and_direct_discard_labels_checked=len(ids),
        actual_normalization_and_tokenizer_menu_order_verified=True, old_and_cross_split_duplicate_inputs=0,
        reserved_groups_excluded=675, no_policy_or_backbone_loaded=True, student_updates=0,
        existing_guard_exclusions_preserved_not_silently_censored=True, training_and_execution_route_admission_pending=True,
        active_fit_dependencies_unchanged=len(active), raw_inputs_labels_and_traces_private=True,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(), goal_achieved=False)
    write(out/'summary-private.json', result); print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'counts')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'out', 'terminal', 'exit'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); audit(a.source.resolve(), a.out.resolve(), a.terminal.resolve(), a.exit.resolve())
