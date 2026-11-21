"""Check real attack replay, ancestral splits, masked gradients and epoch order."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_attack_losses import PREFIX, canonical_question, loss, mixture
from tools.curriculum_course_order import CourseEpochRandom
from tools.curriculum_parent import read, rows, sha, require_child_input_contract, require_replay
from tools.curriculum_transfer import canonical_protocol


def check(registered, data_ready, out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    reg = read(registered)
    data = read(data_ready)
    require_child_input_contract(reg, data)
    for p, h in reg['inputs'].items():
        assert sha(p) == h, p
    for spec in data['files'].values():
        assert sha(ROOT/spec['path']) == spec['sha256']
    sys.path.insert(0, reg['source'])
    sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(1)
    from jevmahjong.training.base_lora import package_digest, question_loss as frozen_loss
    from jevmahjong.training.initialization import check_groups, lineage_training_files
    from jevany.data import api_request
    assert package_digest(Path(reg['source'])/'jevmahjong') == reg['source_sha256']
    fit = ROOT/reg['curriculum']['out']
    protocol = canonical_protocol(read(fit/'protocol.json'))
    lineage = lineage_training_files(protocol)
    train, held = [rows(ROOT/data['files'][split]['path']) for split in ('train', 'validation')]
    group_proof = check_groups(lineage[0], ROOT/data['files']['train']['path'], ROOT/data['files']['validation']['path'], lineage[1:])
    replay_counts = {}
    for key, child in (('data', train), ('eval_data', held)):
        original = rows(protocol['trainer_args'][key])
        require_replay(original, child)
        index = {r['_meta']['id']: r for r in child}
        assert len(index) == len(child)
        for row in original:
            assert api_request(row) == api_request(index[row['_meta']['id']])
        replay_counts[key] = len(original)
    recipe = mixture(train)
    assert recipe['counts'] == data['categories']['train'] and recipe['example_weights'] == data['example_weights']
    assert recipe['loss_mass'] == data['loss_mass']
    strata = [r['questions']['action']['src'].removeprefix(PREFIX) for r in train]
    assert Counter(strata) == Counter(recipe['counts'])
    selected = defaultdict(list)
    for row in train:
        category = row['questions']['action']['src'].removeprefix(PREFIX)
        if len(selected[category]) < 16:
            selected[category].append(row)
    maximum = 0.
    masked_unknown = 0
    for part in selected.values():
        for row in part:
            raw = row['questions']['action']
            keys = list(raw['criteria'])
            q = dict(src=raw['src'], label=keys.index(raw['label']))
            if raw.get('target') is not None:
                q['target'] = [raw['target'].get(k, 0.) for k in keys]
            category, canonical = canonical_question(q)
            a = torch.linspace(-1., 1., len(keys), requires_grad=True)
            b = a.detach().clone().requires_grad_()
            left = loss(a, q, 'cpu', recipe['example_weights'])
            right = frozen_loss(b, canonical, 'cpu')*recipe['example_weights'][category]
            left.backward()
            right.backward()
            maximum = max(maximum, abs(left.item()-right.item()), (a.grad-b.grad).abs().max().item())
            assert torch.equal(left, right) and torch.equal(a.grad, b.grad)
            if 'target' in q:
                unknown = torch.tensor(q['target']) == 0
                assert torch.count_nonzero(a.grad[unknown]).item() == 0
                masked_unknown += int(unknown.any())
    rng = CourseEpochRandom(2026112105, strata, data['count_only_accumulation_proposal']['accumulation'],
        out.parent/'order-precheck-private')
    order = list(range(len(train)))
    for _ in range(2):
        rng.shuffle(order)
        assert Counter(strata[i] for i in order) == Counter(recipe['counts'])
    result = dict(status='attack_course_real_data_and_frozen_gradient_checks_passed', node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'], training_records=len(train), validation_records=len(held),
        parent_replay_records=replay_counts, replay_actual_api_requests_identical=True,
        group_proof=group_proof, ancestor_training_files=lineage,
        real_loss_gradient_comparisons=sum(map(len, selected.values())),
        all10_strata_sampled=True, maximum_loss_gradient_difference=maximum,
        sampled_masked_unknown_menus=masked_unknown, diagnostic_epochs=2,
        count_only_accumulation=data['count_only_accumulation_proposal'],
        data_ready_sha256=sha(data_ready), parent_registration_sha256=sha(registered),
        optimizer_updates=0, student_model_loaded=False, parent_seal_created=False,
        positive_qualified_parent_load_still_pending=True, current_fits_modified=False)
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('registered', 'data-ready', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    check(a.registered.resolve(), a.data_ready.resolve(), a.out.resolve())
