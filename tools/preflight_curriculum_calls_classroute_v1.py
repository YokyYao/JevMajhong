"""Actual13-source packed inputs, CUDA losses/gradients and cached metrics."""
import argparse
from collections import Counter
import copy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write, MASS, POST
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_classroute_v1 import annotate, parts, weighted_loss, summarize, profile, require_recipe, COMPONENTS
from tools.register_curriculum_calls_classroute_v1 import PUBLIC_FILES


def preflight(out):
    gpu_only(); out.mkdir(parents=True, exist_ok=False)
    reg = read(ROOT/'runs/curriculum-riichi-class-control-v1/fit-plan/registered.json')
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from jevany.data import materialize
    from jevany.model import DecisionModel, load_preprocessor
    from jevmahjong.training.base_lora import target_cross_entropy
    from tools.curriculum_calls_riichi_retention_v1 import parts as old_parts
    from tools.curriculum_riichi_control import cached_parts
    files = {str((ROOT/p).resolve()): sha(ROOT/p) for p in PUBLIC_FILES}
    binding_path = ROOT/'runs/curriculum-calls-post-reach-data-v1/completion-binding-private.json'
    files.update(read(binding_path)['files']); files[str(binding_path)] = sha(binding_path)
    data_path = ROOT/'runs/curriculum-calls-post-reach-data-v1/data/data-ready-private.json'; data = read(data_path)
    config = read(ROOT/'configs/training/curriculum-calls-classroute-v1.json'); config['loss_mass'] = MASS
    require_recipe(config, data)
    for name in ('tools.train_curriculum_calls_classroute_v1', 'tools.audit_curriculum_calls_classroute_v1',
                 'tools.evaluate_curriculum_calls_classroute_v1'): importlib.import_module(name)
    assert all(sha(p) == h for p, h in files.items())
    tokenizer = load_preprocessor(reg['training_config']['base_load_path'], backbone_adapter='text')
    proxy = SimpleNamespace(decision_mode='pointer', option_isolation=False)
    representatives = {}; counts = {}; maximum = {}; validation = []
    for split, spec in data['files'].items():
        path = ROOT/spec['path']; assert sha(path) == spec['sha256']; files[str(path)] = spec['sha256']
        counted = Counter(); max_packed = 0
        for line in path.open():
            row = json.loads(line); raw = materialize(row); encoded = annotate(copy.deepcopy(raw), row)
            a = DecisionModel.encode(proxy, tokenizer, raw, max_state=4096, max_branch=8192, strict=True)
            b = DecisionModel.encode(proxy, tokenizer, encoded, max_state=4096, max_branch=8192, strict=True)
            assert a == b and not a['state_truncated'] and len(a['ids']) <= 8192
            question = encoded['questions'][0]; source = question['src'].removeprefix('curriculum2:')
            counted[source] += 1; max_packed = max(max_packed, len(a['ids']))
            representatives.setdefault((split, source), question)
            if split == 'validation': validation.append(row)
            if sum(counted.values()) % 10000 == 0: print(json.dumps(dict(split=split, checked=sum(counted.values()))), flush=True)
        assert counted == data['categories'][split]; counts[split] = dict(counted); maximum[split] = max_packed
    gradients = 0; error = 0.
    for (split, source), question in representatives.items():
        for dtype in (torch.float64, torch.float32):
            z = torch.linspace(-.7, .9, len(question['keys']), device='cuda', dtype=dtype, requires_grad=True)
            y = z.detach().clone().requires_grad_(); observed = []
            actual = weighted_loss(z, question, 'cuda', data['example_weights'], observed)
            if source == POST:
                expected = torch.nn.functional.cross_entropy(y[None], torch.tensor([question['label']], device='cuda'))
            else: expected = old_parts(y, question, 'cuda')['objective']
            expected = expected*data['example_weights'][source]
            e = max(abs(float(actual.detach())-float(expected.detach())),
                float((torch.autograd.grad(actual, z)[0]-torch.autograd.grad(expected, y)[0]).abs().max()))
            assert e < (1e-12 if dtype == torch.float64 else 1e-6) and len(observed) == 1
            gradients += 1; error = max(error, e)
    predictions = []; component_error = 0.; pending = []
    for i, row in enumerate(validation):
        q = annotate(materialize(row), row)['questions'][0]; source = q['src'].removeprefix('curriculum2:')
        z = torch.linspace(-1.1, .7, len(q['keys']), device='cuda', dtype=torch.float64, requires_grad=True)
        if i % 2: z = -z
        p = z.detach().softmax(0).cpu().tolist(); predictions.append(dict(id=row['_meta']['id'], keys=q['keys'], p=p))
        weighted_loss(z, q, 'cuda', data['example_weights'], pending)
        actual = parts(z, q, 'cuda'); raw = row['questions']['action']
        if source.startswith('basic:'):
            expected = dict(cached_parts(dict(raw, src='curriculum1:'+source.removeprefix('basic:')), q['keys'], p, target_cross_entropy), advanced_action_ce=0., post_reach_action_ce=0.)
        else:
            ce = target_cross_entropy(raw, q['keys'], p); expected = {k: 0. for k in COMPONENTS}
            expected.update(objective=ce, standard_action_ce=ce)
            expected['post_reach_action_ce' if source == POST else 'advanced_action_ce'] = ce
        component_error = max(component_error, max(abs(float(actual[k].detach())-v) for k, v in expected.items()))
    summary = summarize(pending, data['example_weights'])
    actual = profile(validation, predictions, MASS, {'weighted_target_ce': 123.}, target_cross_entropy)
    aggregate_error = max(abs(actual['source_loss_components'][s][k]-summary['source_loss_components'][s][k]) for s in MASS for k in COMPONENTS)
    assert component_error < 5e-6 and aggregate_error < 5e-6 and len(validation) == 12479
    assert all(sha(p) == h for p, h in files.items())
    write(out/'binding-private.json', dict(files=files))
    result = dict(status='calls13_classroute_full_inputs_CUDA_loss_and_cached_metrics_preflight_passed',
        packed_rows={s: sum(c.values()) for s, c in counts.items()}, max_packed=maximum,
        actual_row_gradient_comparisons=gradients, maximum_value_gradient_error=error,
        synthetic_cached_validation_menus=len(validation), maximum_component_error=component_error,
        maximum_aggregate_error=aggregate_error, all13_source_counts=counts,
        model_weights_loaded=False, synthetic_logits_not_policy_predictions=True, student_optimizer_updates=0)
    write(out/'summary-private.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path, required=True)
    preflight(p.parse_args().out.resolve())
