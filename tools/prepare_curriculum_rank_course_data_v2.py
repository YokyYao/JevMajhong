"""Prepare separate stage4 input-contract data while preserving every3B row."""
from collections import Counter
import copy
import json
from pathlib import Path
import os
import socket
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.curriculum_rank_win_input_v2 import training_record, serving_request, SCHEMA


def prepare():
    gpu_only(); reg_path = ROOT/'runs/curriculum-progression-v1/fit-plan/registered.json'; reg = read(reg_path)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from types import SimpleNamespace
    from jevany.api import SystemOneRequest, to_record
    from jevany.data import api_request
    from jevany.model import DecisionModel, load_preprocessor
    base_path = ROOT/'runs/curriculum-yaku-scenarios-v1/data-v2/data-ready-private.json'; base = read(base_path)
    targets_path = ROOT/'runs/curriculum-rank-win-labels-v1/merged-v1/data-ready-private.json'; targets = read(targets_path)
    verified_path = ROOT/'runs/curriculum-rank-win-input-v1/verification-v2/summary-private.json'; verified = read(verified_path)
    if (targets['status'] != 'actual_rank32_adaptive_soft_targets_generated_input_policy_and_fit_admission_pending'
            or verified['status'] != 'rank_win_native_input_encoder_and_single_model_route_verified_not_student_fit'
            or verified['input_schema'] != SCHEMA or not verified['same_model_and_parameter_bundle']
            or not verified['no_optimizer_updates']
            or verified['counts']['ordinary_validation_inputs_identical'] != 24516):
        raise ValueError('Completed actual labels and verified input/serving interface required')
    bindings = [targets_path.parent/'completion-binding-private.json',
        verified_path.parent/'completion-binding-private.json', base_path.parent/'completion-binding-private.json']
    files = {}
    for path in bindings:
        binding = read(path)
        for p, h in binding['files'].items():
            if p in files and files[p] != h: raise ValueError('Conflicting immutable input')
            if sha(p) != h: raise ValueError('A completed prerequisite changed: '+p)
            files[p] = h
        files[str(path.resolve())] = sha(path)
    for path in (Path(__file__), reg_path, base_path, targets_path, verified_path,
                 ROOT/'cluster/curriculum-rank-course-data-v2.slurm'):
        files[str(path.resolve())] = sha(path)
    out = ROOT/'runs/curriculum-rank-course-data-v1/data-v1'; out.mkdir(parents=True, exist_ok=False)
    tokenizer = load_preprocessor(reg['training_config']['base_load_path'], backbone_adapter='text')
    recipe = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    proxy = SimpleNamespace(decision_mode='pointer', option_isolation=False)
    groups = {}; ids = set(); new_counts = Counter(); counts = {}; maximum = 0; outputs = {}
    all_pools = {}; source_summary = {}
    for split, expected in (('train', 91247), ('validation', 24516)):
        path = ROOT/base['files'][split]['path']; files[str(path.resolve())] = sha(path)
        if files[str(path.resolve())] != base['files'][split]['sha256']: raise ValueError('Original3B replay changed')
        originals = [json.loads(line) for line in path.open()]
        if len(originals) != expected: raise ValueError('Complete original3B replay required')
        fresh = []; target_path = ROOT/targets['files'][split]['path']; files[str(target_path.resolve())] = sha(target_path)
        if files[str(target_path.resolve())] != targets['files'][split]['sha256']: raise ValueError('Merged candidate labels changed')
        for line in target_path.open():
            row = json.loads(line); original = copy.deepcopy(row); normalized = training_record(row)
            q, before = normalized['questions']['action'], row['questions']['action']
            if (row != original or normalized['_meta'] != row['_meta']
                    or any(q[k] != before[k] for k in ('label', 'target', 'src', 'criteria'))
                    or serving_request(normalized) != serving_request(row)):
                raise ValueError('Input projection changed target/menu/lineage or train/serve surface')
            record, mapping = to_record(SystemOneRequest.model_validate(api_request(normalized)))
            encoded = DecisionModel.encode(proxy, tokenizer, record, max_state=recipe['max_state'],
                max_branch=recipe['max_branch'], strict=True)
            if (encoded['state_truncated'] or len(encoded['ids']) > recipe['max_packed']
                    or mapping[0]['keys'] != list(q['criteria'])): raise ValueError('New input truncates or menu changes')
            maximum = max(maximum, len(encoded['ids'])); fresh.append(normalized); new_counts[q['src']] += 1
        if len(fresh) != targets['files'][split]['records']: raise ValueError('Every admitted candidate label required')
        pool = originals+fresh
        groups[split] = {r['_meta']['group_id'] for r in pool}
        for row in pool:
            if row['_meta']['id'] in ids: raise ValueError('Duplicate or cross-split source ID')
            ids.add(row['_meta']['id'])
        all_pools[split] = (originals, fresh)
        counts[split] = dict(original_replay=len(originals), new_rank_win=len(fresh), total=len(pool), groups=len(groups[split]))
        source_summary[split] = dict(Counter(r['questions']['action']['src'] for r in pool))
    if groups['train'] & groups['validation']: raise ValueError('Whole-family split overlap')
    for split, (originals, fresh) in all_pools.items():
        dest = out/f'{split}-replay-and-rank-win-private.jsonl'
        with dest.open('x') as stream:
            for row in originals+fresh: stream.write(json.dumps(row, separators=(',', ':'))+'\n')
        outputs[split] = dict(path=str(dest.relative_to(ROOT)), sha256=sha(dest),
            records=counts[split]['total'], groups=counts[split]['groups'])
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Immutable input changed during preparation')
    result = dict(status='exact3B_replay_plus_rank_win_input_contract_data_prepared_not_fit_admitted',
        input_schema=SCHEMA, files=outputs, counts=counts, sources=source_summary, new_source_counts=dict(new_counts),
        maximum_new_packed_tokens=maximum, all_original_replay_objects_labels_targets_groups_unchanged=True,
        new_labels_targets_and_family_metadata_unchanged=True, exact_training_serving_projection=True,
        masked_two_arm_target_loss_required=True, train_validation_group_overlap=0,
        source_target_quarantine_preserved=targets['quarantined_views'], candidate_teacher_optimality_not_proven=True,
        source_sampling_and_bootstrap_uncertainty_retained=True, non_win_terminal_value_targets_pending=True,
        optimizer_steps=0, parent_and_training_mix_not_registered=True, final_grid_unused=True,
        raw_records_private=True, job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
    write(out/'data-ready-private.json', result)
    write(out/'completion-binding-private.json', dict(files=files))
    write(out/'summary-public.json', {k: v for k, v in result.items() if k not in ('files','sources')})
    print(json.dumps(read(out/'summary-public.json')), flush=True)


if __name__ == '__main__': prepare()

