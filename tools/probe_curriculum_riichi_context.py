"""A fixed tabular diagnostic, never a deployed Jev correction network."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_riichi_context import FIELD, VERSION
from tools.audit_curriculum_riichi_scores import confusion, ranking_summary

COLUMNS = ('self_is_dealer', 'self_score', 'score_rank', 'gap_to_second', 'lead_over_fourth',
           'turn', 'opponent_riichi_count', 'honba', 'riichi_sticks', 'tenpai_discard_options',
           'tenpai_ukeire_min', 'tenpai_ukeire_max', 'safe_tenpai_discard_options',
           'safe_tenpai_ukeire_max', 'value_honor_roles_min', 'value_honor_roles_max',
           'dora_after_discard_min', 'dora_after_discard_max',
           'red_after_discard_min', 'red_after_discard_max')


def read(path):
    return json.loads(Path(path).read_text())


def probe(data_ready, out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    import torch
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    torch.manual_seed(2026112103)
    data = read(data_ready)
    assert data['status'] == 'riichi_context_variant_prepared_not_fit_admitted' and data['schema'] == VERSION
    for name, expected in (('tools/curriculum_riichi_context.py', data['feature_code_sha256']),
                           ('tools/prepare_curriculum_riichi_context.py', data['preparation_code_sha256'])):
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
    pools, vectors, labels, groups = {}, {}, {}, {}
    for split, spec in data['files'].items():
        path = ROOT/spec['path']
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == spec['sha256']
        all_rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        pool = [r for r in all_rows if any(c['type'] == 'reach' for c in r['questions']['action']['criteria'].values())]
        pools[split] = pool
        vectors[split] = [[next(c[FIELD] for c in r['questions']['action']['criteria'].values() if c['type'] == 'reach')[k]
                           for k in COLUMNS] for r in pool]
        labels[split] = [r['questions']['action']['criteria'][r['questions']['action']['label']]['type'] == 'reach' for r in pool]
        groups[split] = {r['_meta']['group_id'] for r in pool}
    assert not groups['train'] & groups['validation']
    medians = [statistics.median(v[i] for v in vectors['train'] if v[i] is not None)
               if any(v[i] is not None for v in vectors['train']) else 0. for i in range(len(COLUMNS))]
    def encode(part):
        return torch.tensor([[float(v if v is not None else medians[i]) for i, v in enumerate(row)]
                             +[float(v is None) for v in row] for row in part], dtype=torch.float32, device='cuda')
    train, val = encode(vectors['train']), encode(vectors['validation'])
    mean, scale = train.mean(0), train.std(0).clamp_min(1.)
    train, val = (train-mean)/scale, (val-mean)/scale
    target = torch.tensor(labels['train'], dtype=torch.float32, device='cuda')
    model = torch.nn.Linear(train.shape[1], 1).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=.01)
    losses = []
    for step in range(400):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(train).flatten(), target)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        probabilities = torch.sigmoid(model(val).flatten()).cpu().tolist()
    ranking = ranking_summary(labels['validation'], probabilities)
    result = dict(status='tabular_riichi_diagnostic_complete_not_jev_training',
        node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'], device=torch.cuda.get_device_name(0),
        data_ready_sha256=hashlib.sha256(Path(data_ready).read_bytes()).hexdigest(),
        feature_schema=VERSION, columns=COLUMNS, missing_flags_added_for_probe_only=True,
        imputation_and_standardization_fitted_on_train_only=True,
        train_records=len(pools['train']), validation_records=len(pools['validation']),
        train_groups=len(groups['train']), validation_groups=len(groups['validation']),
        train_positive=sum(labels['train']), validation_positive=sum(labels['validation']),
        seed=2026112103, probe_steps=400, optimizer='AdamW lr0.03 wd0.01',
        train_loss_first=losses[0], train_loss_final=losses[-1], real_probe_training_losses=losses,
        validation_at_fixed_half=confusion(labels['validation'], [p >= .5 for p in probabilities]),
        validation_auc=ranking['auc'], no_validation_selected_threshold=True,
        no_jev_or_teacher_model_loaded=True, no_checkpoint_saved=True,
        no_jev_training_metrics_modified=True, active_fit_or_policy_unchanged=True,
        new_curriculum_fit_not_admitted=True,
        note='One fixed linear diagnostic on consumed development labels. Its result measures feature-label signal, not causal feature benefit, complete-game value or Jev strength. No validation tuning, correction network, exported weights or production use. Only a separately registered Jev experiment can assess this representation.')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'real_probe_training_losses'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-ready', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    probe(a.data_ready, a.out)
