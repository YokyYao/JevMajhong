"""Independent exact-row, heldout, tokenizer and masked-target union audit."""
import argparse
from collections import Counter
import itertools
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
    scheduler_exit(read(exit_path), terminal.read_text(), ROOT/'cluster/curriculum-progression-union-v3.slurm')
    ready = read(source/'data-ready-private.json'); binding = read(source/'preparation-binding-private.json')
    if ready['status'] != 'progression_union_v3_prepared_not_independently_audited_or_fit_admitted':
        raise ValueError('Actually completed separately versioned v3 union required')
    files = dict(binding['files']); active = dict(binding['active_fit_files'])
    for path in (Path(__file__), ROOT/'cluster/curriculum-progression-union-audit-v1.slurm',
                 ROOT/'docs/experiments/plans/curriculum-progression-union-audit-20261123.md',
                 source/'data-ready-private.json', source/'preparation-binding-private.json', terminal, exit_path):
        files[str(path.resolve())] = sha(path)
    if any(sha(p) != h for p, h in {**files, **active}.items()): raise ValueError('Bound preparation evidence changed')
    registration = ROOT/'runs/curriculum-calls-classroute-v1/fit-plan/registered.json'; reg = read(registration)
    files[str(registration)] = sha(registration)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevmahjong.training.base_lora import package_digest
    from jevany.data import materialize
    from jevany.model import DecisionModel, load_preprocessor
    from tools.audit_curriculum_basic_teacher import fingerprint
    from tools.curriculum_losses import question_loss
    if (package_digest(Path(reg['source'])/'jevmahjong') != reg['source_sha256']
            or package_digest(Path(reg['training_config']['upstream_source'])/'jevany') != reg['training_config']['upstream_source_sha256']):
        raise ValueError('Pinned runtime/preprocessor changed')
    for name in ('audit_curriculum_basic_teacher.py', 'curriculum_losses.py'):
        files[str(ROOT/'tools'/name)] = sha(ROOT/'tools'/name)
    def load(path):
        path = (ROOT/path).resolve(); digest = sha(path)
        if str(path) in files and files[str(path)] != digest: raise ValueError('Bound source identity changed')
        files[str(path)] = digest
        return read(path)
    def rows(spec):
        path = (ROOT/spec['path']).resolve()
        if sha(path) != spec['sha256']: raise ValueError('Source or output data changed')
        files[str(path)] = spec['sha256']; result = [json.loads(line) for line in path.open()]
        if len(result) != spec.get('records', spec.get('rows')): raise ValueError('Row count differs')
        return result
    c = load('configs/diagnostics/curriculum-progression-union-v3.json')
    old = load(c['calls_manifest']); risk = load(c['risk_manifest']); ordinary = load(c['ordinary_manifest'])
    legacy = load(c['legacy_mc_manifest']); clean = load(c['clean_manifest']); reserved = set(load(c['reserved_groups']))
    recipe = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    tokenizer = load_preprocessor(reg['training_config']['base_load_path'], backbone_adapter='text')
    proxy = SimpleNamespace(decision_mode='pointer', option_isolation=False)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(files=files, active_fit_files=active,
        no_policy_or_backbone_loaded=True, independently_compares_original_row_objects_and_materialized_target_masks=True))
    ids = {s: set() for s in ('train', 'validation')}; groups = {s: set() for s in ids}; inputs = {s: set() for s in ids}
    counts = {}; tokens = {}; mask_gradient_probes = set(); total = 0
    for split in ids:
        # Derive MC membership independently from the original full clean data.
        original = {r['_meta']['id']: r for r in rows(clean['stages']['stage3'][split])}
        earlier = {r['_meta']['id']: r for r in rows(clean['stages']['stage2'][split])}
        if any(original.get(k) != r for k, r in earlier.items()): raise ValueError('Original replay ancestry differs')
        additions = {k: r for k, r in original.items() if k not in earlier}
        informative = {k: r for k, r in additions.items() if sum(v > 0 for v in r['questions']['action']['target'].values()) >= 2}
        zero = {k: r for k, r in additions.items() if sum(v > 0 for v in r['questions']['action']['target'].values()) == 1}
        mc = [r for r in rows(legacy['files'][split]) if r['_meta']['label_type'] == 'paired_future_masked']
        if len(mc) != len(informative) or {r['_meta']['id'] for r in mc} != set(informative):
            raise ValueError('Prepared MC membership differs from original informative source')
        for row in mc:
            source_row = informative[row['_meta']['id']]; restored = dict(row, questions=dict(row['questions']))
            restored['questions']['action'] = dict(row['questions']['action'], src=source_row['questions']['action']['src'])
            if restored != source_row or row['questions']['action']['src'] != 'curriculum3:mc:'+row['_meta']['phase']:
                raise ValueError('Actual MC target/mask/metadata changed')
        quarantine = rows(ready['files'][split+'_quarantine'])
        if len(quarantine) != len(zero) or {r['_meta']['id']: r for r in quarantine} != zero:
            raise ValueError('Original singleton-support quarantine not preserved')
        expected = itertools.chain(rows(old['files'][split]), rows(risk['files'][split]), rows(ordinary['files'][split]), mc)
        actual_path = (ROOT/ready['files'][split]['path']).resolve(); files[str(actual_path)] = sha(actual_path)
        if files[str(actual_path)] != ready['files'][split]['sha256']: raise ValueError('Union output hash changed')
        cc = Counter(); lengths = Counter()
        with actual_path.open() as stream:
            for line, source_row in itertools.zip_longest(stream, expected):
                if line is None or source_row is None: raise ValueError('Merged data omitted or added a record')
                row = json.loads(line)
                if row != source_row: raise ValueError('Merged row differs from its exact original source object')
                meta = row['_meta']; q = row['questions']['action']; rid = meta['id']
                if rid in ids[split]: raise ValueError('Repeated source decision ID')
                if 'source_logical_seed' in meta and meta.get('split') != split: raise ValueError('Teacher family split missing or changed')
                ids[split].add(rid); groups[split].add(meta['group_id']); key = fingerprint(row)
                cc['same_split_identical_public_inputs'] += key in inputs[split]; inputs[split].add(key)
                record = materialize(row); question = record['questions'][0]; keys = list(q['criteria'])
                if question['keys'] != keys or question['label'] != keys.index(q['label']) or question['src'] != q['src']:
                    raise ValueError('Serving materialization reordered options or altered target/source')
                if q.get('target') is not None:
                    raw_target = q['target']; norm = sum(raw_target.values()); vector = [raw_target.get(k, 0.)/norm for k in keys]
                    if question['target'] != vector or sum(v > 0 for v in vector) < 2:
                        raise ValueError('Materialization changed informative masked support/probabilities')
                    cc['masked_MC'] += 1; probe = split, meta['phase']
                    if probe not in mask_gradient_probes:
                        logits = torch.linspace(-.4, .6, len(vector), dtype=torch.float64, requires_grad=True)
                        loss = question_loss(logits, dict(question, src='paired_future_masked'), 'cpu'); loss.backward()
                        mask = torch.tensor(vector) > 0
                        if not torch.isfinite(loss) or not torch.isfinite(logits.grad).all() or (logits.grad[~mask] != 0).any():
                            raise ValueError('Unobserved MC alternatives acquired target gradient')
                        mask_gradient_probes.add(probe)
                else: cc['observed_hard_targets'] += 1
                encoded = DecisionModel.encode(proxy, tokenizer, record, max_state=recipe['max_state'], max_branch=recipe['max_branch'], strict=True)
                if encoded['state_truncated'] or len(encoded['ids']) > recipe['max_packed']:
                    raise ValueError('Merged actual tokenizer truncated a source input')
                lengths['max_packed'] = max(lengths['max_packed'], len(encoded['ids']))
                lengths['max_state'] = max(lengths['max_state'], encoded['seg'].count(0))
                cc['rows'] += 1; total += 1
                if total % 5000 == 0: print(json.dumps(dict(stage='independent_progression_union_encoder_audit', checked=total)), flush=True)
        if cc['rows'] != c['expected_rows'][split] or cc['same_split_identical_public_inputs'] != ready['same_split_duplicate_visible_inputs_retained_and_reported'][split]:
            raise ValueError('Full count or duplicate diagnostics differ')
        counts[split] = dict(cc); tokens[split] = dict(lengths)
    if (ids['train'] & ids['validation'] or groups['train'] & groups['validation'] or groups['train'] & reserved
            or inputs['train'] & inputs['validation'] or len(reserved) != 675 or len(mask_gradient_probes) != 8):
        raise ValueError('Merged held-out exclusions or masked phase coverage failed')
    if any(sha(p) != h for p, h in {**files, **active}.items()): raise ValueError('Immutable evidence changed during audit')
    write(out/'completion-binding-private.json', dict(files=files, active_fit_files=active))
    result = dict(status='exact_progression_union_original_rows_encoder_masks_and_split_audit_passed_not_fit_admitted',
        files=ready['files'], checked_exact_source_row_objects=total, counts=counts, tokens=tokens,
        groups={s: len(v) for s, v in groups.items()}, source_teacher_families_reused=256,
        actual_masked_zero_gradient_probes=len(mask_gradient_probes), original_quarantine_preserved=dict(train=136, validation=41),
        original675_reserved_groups_excluded=True, cross_split_ids_groups_and_public_inputs=0,
        no_policy_or_backbone_loaded=True, student_updates=0, active_fit_dependencies_unchanged=len(active),
        validation_rollout_payloads_opened=False, fit_and_loss_mix_admission_pending=True,
        raw_records_private=True, node=socket.gethostname(), job_id=os.environ['SLURM_JOB_ID'])
    write(out/'summary-private.json', result); print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'out', 'terminal', 'exit'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args(); audit(a.source.resolve(), a.out.resolve(), a.terminal.resolve(), a.exit.resolve())
