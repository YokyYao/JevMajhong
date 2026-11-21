"""Operation coverage, real token admission, and frozen-v8 menu diagnostics."""
import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import socket
import sys
import tomllib
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_parent import read, rows, sha
from tools.curriculum_operation_coverage import OPERATIONS, coverage


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def audit(registered, attack_ready, confidence_summary, out):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-', 'login'))
    out.mkdir(parents=True, exist_ok=False)
    reg = read(registered)
    data = read(attack_ready)
    for p, h in reg['inputs'].items():
        assert sha(p) == h, p
    for spec in data['files'].values():
        assert sha(ROOT/spec['path']) == spec['sha256']
    train, held = [rows(ROOT/data['files'][split]['path']) for split in ('train', 'validation')]
    confidence = None
    confidence_files = {}
    if confidence_summary:
        evidence = read(confidence_summary)
        assert evidence['status'] == 'all_current_train_mc_targets_reproduced_from_paired_worlds'
        assert evidence['data_ready_sha256'] == data['source_data_ready_sha256']
        file = confidence_summary.parent/'conditional-confidence-private.jsonl'
        values = rows(file)
        confidence = {r['id']: r for r in values}
        assert len(confidence) == len(values) == evidence['records']
        assert set(confidence) == {r['_meta']['id'] for r in train if r['_meta']['label_type'] == 'paired_future_masked'}
        confidence_files = {str(confidence_summary): sha(confidence_summary), str(file): sha(file)}
    source = dict(expert_train=coverage([r for r in train if r['_meta']['label_type'] == 'expert_replay']),
        expert_validation=coverage([r for r in held if r['_meta']['label_type'] == 'expert_replay']),
        MC_train=coverage([r for r in train if r['_meta']['label_type'] == 'paired_future_masked'], confidence))
    write(out/'operation-coverage-private.json', source)
    print(json.dumps(dict(stage='coverage_complete',daiminkan={k:v.get('daiminkan') for k,v in source.items()})), flush=True)
    sys.path.insert(0, reg['source'])
    sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(2)
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevany import train as trainer
    from jevany.model import DecisionModel, load_preprocessor
    from jevany.data import api_request
    assert package_digest(Path(reg['source'])/'jevmahjong') == reg['source_sha256']
    assert package_digest(Path(reg['training_config']['upstream_source'])/'jevany') == reg['training_config']['upstream_source_sha256']
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    assert args['decision_mode'] == 'pointer' and args['backbone_adapter'] == 'text' and not args.get('option_isolation', False)
    preprocessor = load_preprocessor(reg['training_config']['base_load_path'], backbone_adapter='text')
    proxy = SimpleNamespace(decision_mode='pointer', option_isolation=False)
    lengths = []
    def encode(self, *pos, **kw):
        value = DecisionModel.encode(self, *pos, **kw)
        assert not value['state_truncated']
        lengths.append((len(value['ids']), sum(s == 0 for s in value['seg'])))
        return value
    proxy.encode = MethodType(encode, proxy)
    admissions = {}
    for split, pool in (('train', train), ('validation', held)):
        # The held split is schema/context checking only, never mixed into a
        # real training file. Use the exact upstream admission implementation.
        a = SimpleNamespace(**dict(args, data=data['files'][split]['path'], replay=0))
        lengths.clear()
        admitted, report = trainer.training_requests(a, preprocessor, None, proxy, inputs={'data': pool})
        assert report['rejected_records'] == 0 and [r['_meta']['id'] for r in admitted] == [r['_meta']['id'] for r in pool]
        admissions[split] = dict(report, maximum_packed_tokens=max(v[0] for v in lengths), maximum_state_tokens=max(v[1] for v in lengths),
            schema_check_only=split == 'validation')
        print(json.dumps(dict(stage='real_token_admission_complete',split=split,admitted=len(admitted),maximum_packed=admissions[split]['maximum_packed_tokens'])), flush=True)
    del preprocessor, proxy
    gc.collect()
    write(out/'token-admission-private.json', dict(admissions=admissions,context_limits={k:args[k] for k in ('max_state','max_branch','max_packed')},
        exact_upstream_training_requests_and_DecisionModel_encode=True, encoding_proxy_has_no_language_model=True,
        no_validation_records_used_for_optimizer=True, optimizer_updates=0))
    checkpoint = ROOT/reg['curriculum']['reference_checkpoint']
    identity = bundle(checkpoint)
    assert identity['identity_sha256'] == reg['curriculum']['reference_identity']
    values = [r for r in held if r['_meta']['label_type'] == 'expert_replay'
        and any(v['type'] in OPERATIONS for v in r['questions']['action']['criteria'].values())]
    client = UnifiedPolicyClient(load_model(checkpoint))
    counts = {op: Counter() for op in OPERATIONS}
    group_sets = {op: set() for op in OPERATIONS}
    with (out/'v8-menu-predictions-private.jsonl').open('x') as stream:
        for i, row in enumerate(values, 1):
            q = row['questions']['action']
            probabilities = client.system_one(**api_request(row))['answers']['action']['probabilities']
            assert set(probabilities) == set(q['criteria'])
            chosen = max(probabilities, key=probabilities.__getitem__)
            actual = q['criteria'][chosen]['type']
            truth = q['criteria'][q['label']]['type']
            for op in {v['type'] for v in q['criteria'].values()} & OPERATIONS:
                yes, selected = truth == op, actual == op
                counts[op]['tp' if yes and selected else 'fn' if yes else 'fp' if selected else 'tn'] += 1
                group_sets[op].add(row['_meta']['group_id'])
            stream.write(json.dumps(dict(id=row['_meta']['id'], group_id=row['_meta']['group_id'], probabilities=probabilities))+'\n')
            if i % 200 == 0:
                stream.flush()
                print(json.dumps(dict(stage='reference_v8_menus',complete=i,total=len(values))), flush=True)
    del client
    gc.collect()
    torch.cuda.empty_cache()
    assert bundle(checkpoint) == identity
    result = dict(status='calls_data_token_and_reference_diagnostic_complete',node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID'],
        attack_data_ready_sha256=sha(attack_ready), current_calls_registration_sha256=sha(registered),
        coverage=source, admissions=admissions, reference_menus=len(values), reference_identity=identity['identity_sha256'],
        reference_operation_counts={op:{k:c[k] for k in ('tp','fn','tn','fp')} for op,c in counts.items()},
        reference_operation_groups={op:len(v) for op,v in group_sets.items()}, confidence_files=confidence_files,
        source_sha256={str(Path(__file__).resolve()):sha(__file__),str(ROOT/'tools/curriculum_operation_coverage.py'):sha(ROOT/'tools/curriculum_operation_coverage.py')},
        source_coverage_does_not_establish_optimality=True, validation_diagnostic_may_include_v8_training_groups=True,
        flat_pointer_before_native_guard=True, new_games=0, optimizer_updates=0, active_fit_modified=False,
        note='Existing observed and soft targets are unchanged. Operation masks must have labelled alternatives on both sides to count as contrasts. Bootstrap diagnostics are conditional and selected-action intervals are nominal, not expert certainty. Frozen v8 predictions on the consumed validation are descriptive and may overlap its old training; they are not held-out strength or new teacher labels. Real tokenizer admission uses the upstream text encoder without constructing a language model; validation is checked separately and never mixed into training.')
    write(out/'summary-private.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('coverage','confidence_files','source_sha256')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('registered','attack-ready','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--confidence-summary',type=Path)
    a=p.parse_args()
    audit(a.registered.resolve(),a.attack_ready.resolve(),a.confidence_summary.resolve() if a.confidence_summary else None,a.out.resolve())
