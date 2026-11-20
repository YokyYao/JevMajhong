"""Recover TRAIN-only conditional future uncertainty; never relabel a fit.

Old opponent hands and continuation policies stay fixed. Bootstrap sensitivity
is conditional on these stored worlds, not a public-belief optimality claim.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.trace_curriculum_mc_provenance import content_identity, load_rows, position, read, sha


def outcome(log, hand_index, seat, draws_before):
    index = -1
    draws = 0
    win = risk = ended = False
    speed = 0.
    for event in log:
        kind = event['type']
        if kind == 'start_kyoku':
            index += 1
        if index != hand_index:
            continue
        if kind == 'tsumo':
            draws += 1
        if kind == 'hora':
            if event['actor'] == seat:
                assert not win and draws >= draws_before
                win = True
                speed = .98 ** (draws-draws_before)
            risk |= event['target'] == seat and event['actor'] != seat
        if kind == 'end_kyoku':
            ended = True
            break
    assert ended, 'An intervention hand is incomplete'
    return float(win), speed, float(risk)


def derive(rows, record, seed, worlds, np):
    first = rows[0]
    eligible = first['eligible']
    teacher = first['teacher_action']
    keys = list(record['questions']['action']['criteria'])
    assert len(set(eligible)) == len(eligible) and teacher in eligible
    assert set(r['action'] for r in rows) == set(eligible)-{teacher}
    assert len(rows) == len(eligible)-1
    matrix = np.zeros((worlds, len(eligible), 3), dtype=np.float64)
    controls = first['counterfactual_worlds']
    assert len(controls) == worlds
    for row in rows:
        assert row['eligible'] == eligible and row['teacher_action'] == teacher
        assert row['public_request'] == first['public_request']
        assert row['source_split'] == 'train'
        assert len(row['counterfactual_worlds']) == worlds
        for i, world in enumerate(row['counterfactual_worlds']):
            assert world['future_seed'] == controls[i]['future_seed']
            assert world['control'] == controls[i]['control']
            a = outcome(world['control']['mjai_log'], first['hand_index'], first['episode_key'][1], first['draws_before'])
            b = outcome(world['branch']['mjai_log'], first['hand_index'], first['episode_key'][1], first['draws_before'])
            matrix[i, eligible.index(row['action'])] = np.array(b)-a
    rng = np.random.default_rng(seed)
    mean = matrix[rng.integers(worlds, size=(256, worlds))].mean(axis=1)
    win, fast, dealin = (mean[..., k] for k in range(3))
    utility = win + .25*fast - 1.5*dealin
    compatible = (win > 1e-12) | ((np.abs(win) <= 1e-12) & (fast > 1e-12)) | (dealin < -1e-12)
    logits = np.where((utility > 0) & ~compatible, 0., utility)/.05
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p = (p/p.sum(axis=1, keepdims=True)).mean(axis=0)
    target = record['questions']['action']['target']
    allowed = [j for j, action in enumerate(eligible) if target.get(keys[action], 0.) > 0]
    assert len(allowed) >= 2
    served = np.zeros(len(keys))
    for j in allowed:
        served[eligible[j]] = p[j]/p[allowed].sum()
    error = max(abs(served[i]-target.get(k, 0.)) for i, k in enumerate(keys))
    assert error < 1e-10, ('Stored target does not reproduce', error)
    chosen = max(allowed, key=lambda j: p[j])
    support_utility = utility[:, allowed]
    # Ties split mass rather than claiming all tied actions are certainly best.
    tied = np.isclose(support_utility, support_utility.max(axis=1, keepdims=True), atol=1e-12, rtol=0)
    win_mass = (tied/tied.sum(axis=1, keepdims=True)).mean(axis=0)
    ci = np.quantile(utility[:, chosen], [.025, .975]).tolist()
    return dict(worlds=worlds, bootstrap_replicates=256, target_reproduction_max_error=error,
        teacher_in_served_support=teacher in [eligible[j] for j in allowed],
        target_argmax_is_control=eligible[chosen] == teacher,
        target_argmax_conditional_best_mass=float(win_mass[allowed.index(chosen)]),
        target_argmax_utility_delta_mean=float((matrix[:, chosen, 0]+.25*matrix[:, chosen, 1]-1.5*matrix[:, chosen, 2]).mean()),
        target_argmax_utility_delta_bootstrap95=ci,
        # Zero baseline has zero utility by construction; do not call it a
        # confident improvement merely because no change beats itself.
        noncontrol_positive_lower_bound=eligible[chosen] != teacher and ci[0] > 0)


def audit(data_ready, out):
    assert os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    assert not socket.gethostname().startswith(('lo-', 'login'))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    receipt = read(data_ready)
    spec = receipt['files']['train_kept']
    assert sha(ROOT/spec['path']) == spec['sha256']
    current = [r for r in load_rows(ROOT/spec['path']) if r['_meta']['label_type'] == 'paired_future_masked']
    original_root = ROOT/'runs/unified-attackrisk-data-v8'
    original = read(original_root/'result.json')
    spec_old = original['files']['train']
    assert sha(spec_old['path']) == spec_old['sha256']
    index = {}
    for row in load_rows(spec_old['path']):
        if row['_meta']['label_type'] == 'paired_future_masked':
            key = content_identity(row)
            assert key not in index
            index[key] = row
    wanted = {}
    for row in current:
        old = index[content_identity(row)]
        pos = position(old, old['_meta']['group_id'])
        assert pos not in wanted
        wanted[pos] = row
    assert len(wanted) == len(current) == 2722
    native_seeds = {int(p.split('|')[0].split(':')[1]) for p in wanted}
    assert not native_seeds & set(range(2026300000, 2026300032))
    manifest = read(original_root/'inputs-private.json')
    bound = {}
    def bind(path):
        path = Path(path).resolve()
        value = sha(path)
        assert manifest[str(path)] == value, ('Source not hash-bound', str(path))
        bound[str(path)] = value
        return path
    source = ROOT/'runs/code/13138ac58cbc4ff95f3aafdf813c50f2b0f42c70b989dfe57377ae57e94d7752'
    sys.path.insert(0, str(source))
    from jevmahjong.data.base_lora_pool import normalized_request
    # Hash-check imported sampler/normalizer source against the original
    # package manifest rather than trusting a mutable working-tree module.
    from jevmahjong.training.base_lora import package_digest
    assert package_digest(source/'jevmahjong') == source.name
    pool = ROOT/'runs/unified-winrisk-pool-v2'
    pool_receipt = read(bind(pool/'result.json'))
    database = bind(pool/'pool-private.sqlite')
    with sqlite3.connect('file:'+str(database)+'?mode=ro', uri=True) as db:
        priorities = {p: priority for p, priority in db.execute("SELECT position,priority FROM cases WHERE split='train' AND label_type='paired_future_masked'") if p in wanted}
    additions = read(bind(ROOT/'runs/configs/unified-attack-data-v7.json'))
    sources = [(s, 'legacy') for s in pool_receipt['config']['sources']]
    sources += [(s, 'attack-v7') for s in additions['sources']]
    recovered = {}
    shard_count = 0
    with (out/'conditional-confidence-private.jsonl').open('x') as dest:
        for source_spec, family in sources:
            directory = Path(source_spec['path'])
            protocol_path = directory/'protocol.json'
            if not protocol_path.exists():
                continue
            protocol = read(protocol_path)
            if protocol.get('collection_method') != 'paired_future_ev_v1':
                continue
            bind(protocol_path)
            config = protocol['config']
            for path in sorted((directory/'walls-private').glob('*.pt')):
                seed = int(path.stem)
                if seed not in native_seeds or (seed-config['training_seed_start']) % 4 == 3:
                    continue
                bind(path)
                payload = torch.load(path, map_location='cpu', weights_only=False)
                shard_count += 1
                groups = defaultdict(list)
                for row in payload['rows']:
                    if 'counterfactual_worlds' not in row:
                        continue
                    pos = position(row['public_request'], 'native:'+str(row['episode_key'][0]))
                    if pos in wanted and pos not in recovered:
                        groups[(pos, row['request_sha256'])].append(row)
                for (pos, _), rows in groups.items():
                    if pos in recovered:
                        continue
                    if family == 'legacy' and priorities.get(pos) != source_spec['priority']:
                        continue
                    if family == 'attack-v7' and pos in priorities:
                        continue
                    first = rows[0]
                    public = normalized_request(first['public_request'], 'native:'+str(first['episode_key'][0]), 'paired_future_masked', 0)
                    record = wanted[pos]
                    if public['state'] != record['state'] or public['questions']['action']['criteria'] != record['questions']['action']['criteria']:
                        continue
                    label_seed = int(hashlib.sha256((('2026111511|' if family == 'attack-v7' else '')+pos).encode()).hexdigest()[:16], 16)
                    result = derive(rows, record, label_seed, config['worlds'], np)
                    result.update(id=record['_meta']['id'], group_id=record['_meta']['group_id'], phase=record['_meta']['phase'],
                        source_shard=str(path.relative_to(ROOT)), source_sha256=bound[str(path.resolve())], family=family)
                    recovered[pos] = result
                    dest.write(json.dumps(result)+'\n')
                del payload
                if shard_count % 20 == 0:
                    dest.flush()
                    print(json.dumps(dict(shards=shard_count, recovered=len(recovered), required=len(wanted))), flush=True)
    assert set(recovered) == set(wanted), ('Incomplete source/target audit', len(recovered), len(wanted))
    rows = list(recovered.values())
    phases = {}
    for phase in sorted({r['phase'] for r in rows}):
        part = [r for r in rows if r['phase'] == phase]
        phases[phase] = dict(records=len(part), noncontrol_positive_lower_bound=sum(r['noncontrol_positive_lower_bound'] for r in part),
            control_argmax=sum(r['target_argmax_is_control'] for r in part),
            median_conditional_best_mass=float(np.median([r['target_argmax_conditional_best_mass'] for r in part])))
    result = dict(status='all_current_train_mc_targets_reproduced_from_paired_worlds', node=socket.gethostname(),
        job_id=os.environ['SLURM_JOB_ID'], records=len(rows), source_groups=len({r['group_id'] for r in rows}), shards=shard_count,
        worlds=dict(Counter(r['worlds'] for r in rows)), phases=phases,
        maximum_target_reproduction_error=max(r['target_reproduction_max_error'] for r in rows),
        data_ready_sha256=sha(data_ready), original_manifest_sha256=sha(original_root/'inputs-private.json'), inputs=bound,
        reward=dict(own_win=1., fast_weight=.25, speed_gamma=.98, dealin_penalty=1.5, net_weight=0.),
        bootstrap_replicates=256, temperature=.05, conditional_confidence_only=True,
        new_simulations=0, optimizer_updates=0, labels_changed=False, validation_rollout_payloads_opened=False,
        practical_v8_and_active_fits_modified=False,
        note='Bootstrap256 reuses16/32 stored worlds; it does not add evidence. Opponent concealed hands and old continuation policy remain conditional assumptions. Do not filter all uncertain alternatives into hard positives or claim expert/public-belief labels. Only current TRAIN kept rows were inspected; no new held-out outcomes or final reserved seeds were opened.')
    (out/'summary-private.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'inputs'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-ready', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    audit(args.data_ready.resolve(), args.out.resolve())
