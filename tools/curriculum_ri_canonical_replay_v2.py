"""Use the pinned loader's canonical requests without relaxing admission.

The V1 replay compared raw JSON to the loader's metadata-enriched records.
Keep V1's complete object equality and rejection checks; construct its replay
from independently loaded canonical records instead. No labels or loss change.
"""
from collections import Counter
import json
from pathlib import Path

from tools.curriculum_ri_replay_epoch_adapter_v1 import EpochReplay

ADDED_METADATA = frozenset(('source', 'variant', 'id', 'group_id', 'row', 'split', 'text_sha256'))
METHOD = 'pinned_loader_canonical_requests_before_unchanged_epoch_admission_v2'


def verify_metadata_only(raw, canonical):
    if not raw or len(raw) != len(canonical):
        raise ValueError('All original unique rows required')
    added = Counter()
    changed_rows = 0
    for before, after in zip(raw, canonical):
        if ({k: v for k, v in before.items() if k != '_meta'}
                != {k: v for k, v in after.items() if k != '_meta'}):
            raise ValueError('Canonical loading changed state/questions/labels or another request field')
        old, new = before.get('_meta', {}), after.get('_meta', {})
        if not isinstance(old, dict) or not isinstance(new, dict):
            raise ValueError('Object metadata required')
        if any(k not in new or new[k] != v for k, v in old.items()):
            raise ValueError('Canonical loading changed existing metadata, identity or split')
        extra = set(new) - set(old)
        if not extra <= ADDED_METADATA:
            raise ValueError('Unregistered loader metadata was added')
        added.update(extra)
        changed_rows += bool(extra)
    return dict(method=METHOD, unique_rows=len(raw), metadata_enriched_rows=changed_rows,
        added_metadata_counts=dict(added), all_nonmetadata_fields_exact=True,
        every_existing_metadata_field_exact=True, row_order_unchanged=True,
        actual_token_admission_claimed=False)


def canonical_replay(raw, plans, data_path, loader):
    # Invoke the real pinned loader, not a local reconstruction of its defaults.
    canonical = loader(data_path)
    report = verify_metadata_only(raw, canonical)
    replay = EpochReplay(canonical, plans)
    replay.canonicalization_report = report
    return replay


def preflight():
    import sys
    import tomllib
    from tools.capture_curriculum_paired_terminal_v2 import resource
    from tools.curriculum_transfer import read, sha
    from tools.curriculum_calls_post_reach_data_v1 import write
    root = Path(__file__).resolve().parents[1]
    allocation = resource()
    registration = root/'runs/curriculum-ri-balanced-replay-fit-v1/fit-plan/registered.json'
    reg = read(registration)
    from tools.curriculum_ri_balanced_replay_runtime_v1 import require_registration
    require_registration(reg)
    sys.path[:0] = [reg['source'], reg['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    from jevany import train as trainer
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    path = Path(args['data'])
    raw = [json.loads(line) for line in path.open()]
    plans = [read(p) for p in reg['replay_epoch_plans']]
    replay = canonical_replay(raw, plans, path, trainer.load_records)
    if len(raw) != 87595 or replay.size != 90228:
        raise ValueError('Full frozen TRAIN and physical plan required')
    report = dict(requested_records=len(raw), admitted_records=len(raw), rejected_records=0, rejected=[])
    # This intentionally verifies loader behavior, not tokenizer admission.
    # Formal fit must still perform full real upstream model token admission.
    admitted = trainer.load_records(path)
    pool, expansion = replay.admit(admitted, report)
    if len(pool) != 90228:
        raise ValueError('Canonical all-row equality did not admit the physical plan')
    from tools.curriculum_ri_replay_epoch_adapter_v1 import EpochReplay as Original
    try:
        Original(raw, plans).admit(admitted, report)
    except ValueError:
        reproduced = True
    else:
        raise ValueError('The actual V1 raw/canonical failure did not reproduce')
    out = root/'runs/curriculum-ri-canonical-replay-v2/preflight-v1'
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(replay.canonicalization_report, allocation=allocation,
        status='full87595_loader_metadata_cause_reproduced_and_canonical_equality_passed',
        v1_failure_reproduced=reproduced, model_loaded=False, student_updates=0,
        new_games=0, formal_actual_token_admission_still_required=True,
        physical_slots=len(pool))
    write(out/'summary-private.json', summary)
    files = dict(reg['inputs'])
    for p in (registration, Path(__file__).resolve(),
              root/'tests/core/test_curriculum_ri_canonical_replay_v2.py',
              Path(trainer.__file__).resolve(), out/'summary-private.json'):
        files[str(p)] = sha(p)
    write(out/'completion-binding-private.json', dict(files=files))
    print(json.dumps({k: v for k, v in summary.items() if k != 'allocation'}), flush=True)


if __name__ == '__main__':
    preflight()
