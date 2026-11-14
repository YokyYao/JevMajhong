"""Average paired-future target distributions before fitting a unified policy."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from jevmahjong.core.io import digest, write
from .base_lora_pool import position
from .relabel_targets import bind, materialize

POLICY = 'paired_bootstrap_average_masked_v1'


def bootstrap_probabilities(deltas, *, seed, temperature=.05, replicates=256):
    """Resample whole paired futures, never separate actions or outcome fields.

    Axes are future, action, and (utility, own_win, fast_reward, dealin).
    This estimates label sensitivity conditional on stored futures; it does
    not add observations or resolve continuation-policy / hidden-hand bias.
    """
    values = np.asarray(deltas, dtype=np.float64)
    if (values.ndim != 3 or values.shape[0] < 2 or values.shape[1] < 1
            or values.shape[2] != 4 or not np.isfinite(values).all()):
        raise ValueError('At least two finite paired futures and four outcome fields required')
    if (not math.isfinite(temperature) or temperature <= 0
            or isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1):
        raise ValueError('Positive finite temperature and integer replicate count required')
    rng = np.random.default_rng(seed)
    indices = rng.integers(values.shape[0], size=(replicates, values.shape[0]))
    means = values[indices].mean(axis=1)
    u, w, f, d = (means[..., k] for k in range(4))
    compatible = (w > 1e-12) | ((np.abs(w) <= 1e-12) & (f > 1e-12)) | (d < -1e-12)
    logits = np.where((u > 0) & ~compatible, 0., u) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    raw = np.exp(logits)
    probabilities = raw / raw.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or (probabilities <= 0).any():
        raise ValueError('Bootstrap labelled support underflowed or became nonfinite')
    return probabilities


def prepare(cfg):
    """Apply a complete, hash-bound training-only bootstrap derivation."""
    if cfg['policy'] != POLICY:
        raise ValueError('Explicit paired bootstrap policy required')
    inputs = {}
    parent_path = bind(inputs, cfg['parent_data_receipt'], cfg['parent_data_receipt_sha256'])
    parent = json.loads(parent_path.read_text())
    if parent['status'] != 'balanced_raw_data_ready' or not parent['requires_masked_soft_target_loss']:
        raise ValueError('Admitted masked training data required')
    for record in parent['files'].values():
        bind(inputs, record['path'], record['sha256'])
    audit_path = bind(inputs, cfg['audit_receipt'], cfg['audit_receipt_sha256'])
    audit = json.loads(audit_path.read_text())
    if (audit['status'] != 'passed_complete_training_only_stability_audit'
            or audit['held_rollout_payloads_opened'] or audit['current_evaluation_game_payloads_opened']
            or audit['optimizer_updates'] or audit['new_simulations'] or audit['bootstrap_replicates'] != 256):
        raise ValueError('Complete train-only audit without optimizer or new simulation required')
    for path, checksum in audit['inputs'].items():
        bind(inputs, path, checksum)
    train = Path(parent['files']['train']['path']).resolve()
    if audit['inputs'].get(str(train)) != parent['files']['train']['sha256']:
        raise ValueError('Audit was not on the actual admitted training bytes')
    target_path = bind(inputs, cfg['bootstrap_targets'], audit['bootstrap_targets_sha256'])
    targets = json.loads(target_path.read_text())
    kernel_path = bind(inputs, cfg['kernel_verification_receipt'], cfg['kernel_verification_receipt_sha256'])
    proof = json.loads(kernel_path.read_text())
    if (proof['status'] != 'passed_all_admitted_bootstrap_kernel_targets'
            or proof['states_checked'] != audit['states_checked']
            or proof['bootstrap_targets_sha256'] != audit['bootstrap_targets_sha256']
            or proof['kernel_source_sha256'] != digest(Path(__file__))):
        raise ValueError('Actual paired-future kernel reproduction required')
    for path, checksum in proof['inputs'].items():
        bind(inputs, path, checksum)
    rows = [json.loads(line) for line in train.read_text().splitlines() if line.strip()]
    revisions = {}; used = set(); phases = Counter(); changed = Counter(); argmax = Counter()
    for row in rows:
        if row['_meta']['label_type'] != 'paired_future_masked':
            continue
        pos = position(row, row['_meta']['group_id'])
        key = hashlib.sha256(pos.encode()).hexdigest()
        item = targets[key]; q = row['questions']['action']; old = q['target']
        if (key in used or item['position'] != pos or item['phase'] != row['_meta']['phase']
                or set(item['full_target']) != set(old)
                or any(abs(item['full_target'][k] - old[k]) > 1e-12 for k in old)):
            raise ValueError('Bootstrap labels differ from unique admitted position or original target')
        proposed = {k: item['target'].get(k, 0.) for k in old}
        if (set(item['target']) != {k for k, v in old.items() if v > 0}
                or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in proposed.values())
                or abs(sum(proposed.values()) - 1.) > 1e-10
                or {k for k, v in proposed.items() if v > 0} != {k for k, v in old.items() if v > 0}):
            raise ValueError('Bootstrap changed labelled support or normalization')
        revisions[pos] = proposed; used.add(key); phase = row['_meta']['phase']; phases[phase] += 1
        changed[phase] += max(abs(proposed[k] - old[k]) for k in old) > 1e-12
        argmax[phase] += max(old, key=old.get) != max(proposed, key=proposed.get)
    if used != set(targets) or len(used) != audit['states_checked']:
        raise ValueError('Bootstrap omitted or introduced training positions')
    for path, checksum in inputs.items():
        if digest(path) != checksum:
            raise ValueError('Input changed before materialization')
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=False)
    files = materialize(parent, revisions, out)
    for path, checksum in inputs.items():
        if digest(path) != checksum:
            raise ValueError('Input changed during materialization')
    stats = dict(states_checked=len(used), phase_states=dict(phases), affected_states=dict(changed), argmax_changes=dict(argmax))
    result = copy.deepcopy(parent)
    result.update(config=cfg, files=files, training_target_policy=POLICY, bootstrap_relabel=stats,
        training_position_order_and_IDs_preserved=True, expert_labels_changed=False,
        validation_targets_changed=False, unknown_actions_are_not_negative_labels=True,
        raw_records_public=False, optimizer_updates=0, new_simulations=0)
    result['fixed_full_validation'] = dict(path=parent['files']['validation']['path'],
        sha256=files['validation']['sha256'], records=files['validation']['records'], bytes_identical=True)
    write(out / 'inputs-private.json', inputs)
    result['target_derivation_inputs_sha256'] = digest(out / 'inputs-private.json')
    write(out / 'result.json', result)
    write(out / 'status.json', dict(status='balanced_raw_data_ready', target_policy=POLICY, **stats))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    result = prepare(json.loads(parser.parse_args().config.read_text()))
    print(json.dumps(dict(status=result['status'], files=result['files'], bootstrap_relabel=result['bootstrap_relabel'])))


if __name__ == '__main__':
    main()
