"""Prepare, but do not admit or fit, a public-riichi-context input variant."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_riichi_context import FIELD, VERSION, enrich_request


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def visible(row):
    return dict(state=row['state'], questions={k: {a: b for a, b in q.items()
        if a in ('type', 'instructions', 'criteria')} for k, q in row['questions'].items()})


def prepare(registered, out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    reg = read(registered)
    original_ready = ROOT/reg['curriculum']['data_root']/'data-ready.json'
    if sha(original_ready) != reg['inputs'][str(original_ready.resolve())]:
        raise ValueError('Frozen source data manifest changed')
    data = read(original_ready)
    prepared, summaries, group_sets = {}, {}, {}
    for split, spec in data['stages']['stage1'].items():
        path = ROOT/spec['path']
        assert sha(path) == spec['sha256']
        pool = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        output, counts, missing = [], Counter(), Counter()
        for row in pool:
            enriched = enrich_request(row)
            oldq, newq = row['questions']['action'], enriched['questions']['action']
            assert oldq['label'] == newq['label'] and oldq.get('target') == newq.get('target')
            assert set(oldq['criteria']) == set(newq['criteria'])
            assert row['_meta'] == enriched['_meta']
            reach = [k for k, c in oldq['criteria'].items() if c['type'] == 'reach']
            if reach:
                counts['reach_windows'] += 1
                counts['teacher_accept' if oldq['criteria'][oldq['label']]['type'] == 'reach' else 'teacher_decline'] += 1
                assert visible(enriched) == enrich_request(visible(row))
                altered = copy.deepcopy(row)
                altered['_meta'] = dict(hidden_future_value=99999)
                altered['questions']['action'].update(label=next(k for k in oldq['criteria'] if k != oldq['label']),
                                                       src='counterfactual-label-only', target={oldq['label']: 1.})
                assert visible(enrich_request(altered)) == visible(enriched)
                for key in reach:
                    context = newq['criteria'][key][FIELD]
                    counts['complete_tenpai_summaries'] += context['complete_tenpai_summary']
                    counts['windows_facing_riichi'] += context['opponent_riichi_count'] > 0
                    counts['value_honor_material_present'] += (context['value_honor_roles_max'] or 0) > 0
                    missing.update(k for k, v in context.items() if v is None)
            else:
                assert visible(enriched) == visible(row)
                counts['unchanged_plain_windows'] += 1
            output.append(enriched)
        assert len({r['_meta']['id'] for r in output}) == len(output)
        group_sets[split] = {r['_meta']['group_id'] for r in output}
        prepared[split] = output
        summaries[split] = dict(records=len(output), source_groups=len(group_sets[split]),
            counts=dict(counts), missing_context_fields=dict(missing),
            label_metadata_counterfactual_invariance_verified=True,
            training_and_public_inference_views_equal=True, no_action_selected_by_transform=True)
    reserved_file = original_ready.parent/'reserved-public-groups.json'
    assert not group_sets['train'] & (group_sets['validation'] | set(read(reserved_file)))
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for split, pool in prepared.items():
        path = out/f'{split}.jsonl'
        with path.open('x') as f:
            for row in pool:
                f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n')
        files[split] = dict(path=str(path), records=len(pool), groups=len(group_sets[split]), sha256=sha(path))
    result = dict(status='riichi_context_variant_prepared_not_fit_admitted', schema=VERSION, field=FIELD,
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'], files=files, summaries=summaries,
        source_data_ready=str(original_ready), source_data_ready_sha256=sha(original_ready),
        source_files=data['stages']['stage1'], feature_code_sha256=sha(ROOT/'tools/curriculum_riichi_context.py'),
        preparation_code_sha256=sha(Path(__file__)), stage1_strata=data['stage1_strata'],
        stage1_loss_mass=data['stage1_loss_mass'], stage1_loss_weights=data['stage1_loss_weights'],
        reserved_group_file=str(reserved_file), reserved_group_sha256=sha(reserved_file),
        all_original_rows_once=True, copies_added=0, labels_targets_and_groups_unchanged=True,
        public_information_only=True, no_teacher_action_or_future_used_in_features=True,
        active_fit_modified=False, new_model_or_optimizer_loaded=False,
        variant_fit_admitted=False, later_course_input_contracts_not_admitted=True,
        note='Preparation is a candidate input representation, not evidence of improved Jev policy. The transform summarizes existing public observations/candidate features; no action is forced. A new fit requires explicit input-version registration and the same transform at native/model inference. Do not enable it on the current frozen fit or practical v8.')
    with (out/'data-ready.json').open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    prepare(a.registered, a.out)
