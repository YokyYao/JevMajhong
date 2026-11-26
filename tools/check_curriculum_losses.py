"""Verify the public loss module against actual frozen losses on real records."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_losses import question_loss, weighted_loss


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def check(registered, data_ready, out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    import torch
    torch.set_num_threads(1)
    reg = read(registered)
    sys.path.insert(0, reg['source'])
    from jevmahjong.training.base_lora import package_digest, question_loss as frozen_loss
    from jevmahjong.training.label_balance import weighted_loss as frozen_weighted_loss
    assert package_digest(Path(reg['source'])/'jevmahjong') == reg['source_sha256']
    data = read(data_ready)
    for spec in data['files'].values():
        with (ROOT/spec['path']).open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == spec['sha256']
    pool = rows(ROOT/data['files']['train_kept']['path'])
    sample = [r for r in pool if r['_meta']['label_type'] == 'expert_replay'][:32]
    sample += [r for r in pool if r['_meta']['label_type'] == 'paired_future_masked'][:128]
    zero = rows(ROOT/data['files']['train_quarantined']['path'])
    sample += zero
    weights = data['loss_proposal']['label_source_weights']
    phase_weights = data['loss_proposal']['MC_phase_weights']
    maximum = 0.
    for row in sample:
        raw = row['questions']['action']
        keys = list(raw['criteria'])
        q = dict(label=keys.index(raw['label']), src=raw.get('src', 'expert_replay'))
        if raw.get('target') is not None:
            q['target'] = [raw['target'].get(k, 0.) for k in keys]
        for phase_weighted in (False, True):
            actual = dict(q)
            if phase_weighted and q['src'] == 'paired_future_masked':
                actual['src'] += ':'+row['_meta']['phase']
            a = torch.linspace(-1., 1., len(keys), requires_grad=True)
            b = a.detach().clone().requires_grad_()
            left = weighted_loss(a, actual, 'cpu', weights, phase_weights) if phase_weighted else question_loss(a, actual, 'cpu')
            right = frozen_weighted_loss(b, actual, 'cpu', weights, phase_weights) if phase_weighted else frozen_loss(b, actual, 'cpu')
            left.backward()
            right.backward()
            maximum = max(maximum, abs(left.item()-right.item()), (a.grad-b.grad).abs().max().item())
            assert torch.equal(left, right) and torch.equal(a.grad, b.grad)
            if raw.get('target') is not None and sum(v > 0 for v in raw['target'].values()) == 1:
                assert left.item() == 0. and torch.count_nonzero(a.grad).item() == 0
    result = dict(status='public_and_frozen_loss_gradient_parity_passed', node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'], real_records=len(sample), singleton_records=len(zero),
        actual_loss_comparisons=2*len(sample), maximum_loss_or_gradient_difference=maximum,
        frozen_package=reg['source_sha256'], model_or_optimizer_loaded=False,
        active_fit_inputs_changed=False)
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered', type=Path, required=True)
    p.add_argument('--data-ready', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    check(a.registered, a.data_ready, a.out)
