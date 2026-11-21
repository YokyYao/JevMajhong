"""Prepare replay-preserving attack/risk inputs; no parent or fit admission."""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_attack_losses import PREFIX, mixture, stratum
from tools.curriculum_parent import read, rows, sha, require_replay
from tools.audit_curriculum_accumulation import suggest_accumulation
from tools.build_curriculum_stage2 import visible


def prepare(data_ready, replay_ready, reserved_groups, out, world_audit=None):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    data = read(data_ready)
    replay = read(replay_ready)
    pools = {}
    for split in ('train', 'validation'):
        spec = data['files'][split+'_kept']
        assert sha(ROOT/spec['path']) == spec['sha256']
        pools[split] = rows(ROOT/spec['path'])
        prior = replay['files'][split]
        assert sha(ROOT/prior['path']) == prior['sha256']
        require_replay(rows(ROOT/prior['path']), pools[split])
    train_groups = {r['_meta']['group_id'] for r in pools['train']}
    held_groups = {r['_meta']['group_id'] for r in pools['validation']}
    reserved = set(read(reserved_groups))
    assert not train_groups & (held_groups | reserved)
    train_ids = {r['_meta']['id'] for r in pools['train']}
    held_ids = {r['_meta']['id'] for r in pools['validation']}
    assert len(train_ids) == len(pools['train']) and len(held_ids) == len(pools['validation'])
    assert not train_ids & held_ids
    audit_identity = None
    if world_audit:
        audit = read(world_audit)
        assert audit['status'] == 'all_current_train_mc_targets_reproduced_from_paired_worlds'
        assert audit['data_ready_sha256'] == sha(data_ready)
        assert audit['records'] == sum(r['_meta']['label_type'] == 'paired_future_masked' for r in pools['train'])
        assert audit['maximum_target_reproduction_error'] < 1e-10
        assert not audit['labels_changed'] and not audit['optimizer_updates']
        audit_identity = dict(path=str(world_audit), sha256=sha(world_audit), records=audit['records'],
            conditional_confidence_only=True, new_target_weighting_from_confidence=False)
    recipe = mixture(pools['train'])
    order = suggest_accumulation(len(pools['train']), 128, min(recipe['counts'].values()))
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for split, pool in pools.items():
        tagged = []
        for source in pool:
            row = copy.deepcopy(source)
            row['questions']['action']['src'] = PREFIX+stratum(source)
            assert visible(row) == visible(source)
            tagged.append(row)
        require_replay(pool, tagged)
        path = out/(split+'.jsonl')
        with path.open('x') as f:
            for row in tagged:
                f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n')
        files[split] = dict(path=str(path), sha256=sha(path), records=len(pool),
            groups=len({r['_meta']['group_id'] for r in pool}))
    result = dict(status='stage3_attack_data_prepared_pending_parent_and_fit_registration',
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'], files=files,
        required_input_schema=data.get('required_input_schema'),
        source_data_ready=str(data_ready), source_data_ready_sha256=sha(data_ready),
        replay_data_ready=str(replay_ready), replay_data_ready_sha256=sha(replay_ready),
        reserved_groups=str(reserved_groups), reserved_groups_sha256=sha(reserved_groups),
        source_world_audit=audit_identity, categories={k: mixture(v)['counts'] for k, v in pools.items()},
        loss_mass=recipe['loss_mass'], example_weights=recipe['example_weights'],
        count_only_accumulation_proposal=order, informative_rows_unchanged=True,
        replay_train_and_validation_exact=True, model_visible_inputs_unchanged=True,
        targets_labels_groups_and_IDs_unchanged=True, copies_added=0,
        train_validation_and_reserved_group_overlap=0, conditional_uncertainty_not_filtered=True,
        masked_target_identity_must_be_restored=True, optimizer_updates=0, parent_admitted=False,
        note='75% expert replay preserving the calls-course relative loss mixture;25% conditional MC, equally weighted across four phases. Loss-only tags never enter model input. This preparation changes neither labels nor active fits. Registration still requires an audited final calls parent, exact initialization, a new optimizer and a separately frozen budget. An absent private world audit is disclosed, not fabricated; the public fixed labels can still be reproduced.')
    (out/'data-ready.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('data-ready', 'replay-ready', 'reserved-groups', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--world-audit', type=Path)
    a = p.parse_args()
    prepare(a.data_ready.resolve(), a.replay_ready.resolve(), a.reserved_groups.resolve(), a.out.resolve(),
        a.world_audit.resolve() if a.world_audit else None)
