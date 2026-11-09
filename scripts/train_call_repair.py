"""Frozen-backbone, query-only call repair with grouped and sealed holdouts."""
import argparse
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from call_policy import digest, is_call_request, metrics, request_hash, screenshot_view, CallRouter
from head_rl import write


def make_head(meta, device):
    from jevany.model import PointerHead
    state = meta['head']
    if meta.get('temperature') != 1. or meta.get('head_residual_dim') != 0:
        raise ValueError('Expected plain temperature1 pointer')
    dp, d = state['q.weight'].shape
    head = PointerHead(d, dp).to(device)
    head.load_state_dict(state)
    return head


def payload(record, view):
    rec = record if view == 'native' else screenshot_view(record, unknown=view == 'screen_unknown')
    value = {'state': copy.deepcopy(rec['state']), 'questions': copy.deepcopy(rec['questions'])}
    value['questions']['action'].pop('label', None)
    return value


def logits_batch(head, rows, device):
    decide = torch.stack([r['decide'] for r in rows]).to(device)
    count = max(len(r['options']) for r in rows)
    options = torch.zeros(len(rows), count, decide.shape[-1], device=device)
    mask = torch.zeros(len(rows), count, dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        n = len(row['options']); options[i, :n] = row['options'].to(device); mask[i, :n] = True
    logits = (head.k(options) * head.q(decide)[:, None, :]).sum(-1) * head.scale
    return logits.masked_fill(~mask, float('-inf'))


@torch.no_grad()
def evaluate(head, rows, device, batch_size=64):
    results = {}
    for view in sorted({r['view'] for r in rows}):
        selected = [r for r in rows if r['view'] == view]
        scored = []
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            probabilities = logits_batch(head, batch, device).softmax(-1).cpu()
            for row, probability in zip(batch, probabilities):
                pred = int(probability.argmax()); gold = row['label']
                scored.append({'gold': row['types'][gold], 'pred': row['types'][pred],
                    'exact': pred == gold, 'target_probability': float(probability[gold])})
        results[view] = metrics(scored)
    return {'views': results, 'macro_f1': sum(v['macro_f1'] for v in results.values()) / len(results),
            'nll': sum(v['nll'] for v in results.values()) / len(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    data, out, reference = (ROOT / cfg[k] for k in ('data', 'out', 'reference'))
    audit = json.loads((data / 'provenance.json').read_text())
    if not audit['complete'] or audit['counts']['train'] != sum(cfg['train_quotas'].values()):
        raise ValueError('Data not complete')
    for split, h in audit['files'].items():
        if digest(data / f'{split}.jsonl') != h: raise ValueError('Data changed after audit')
    if digest(reference / 'head.pt') != cfg['reference_head_sha256']:
        raise ValueError('Confirmed reference changed')
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if set(audit['matches'][a]) & set(audit['matches'][b]): raise ValueError('Split overlap')
    if any(audit['action_types'][s].get(k, 0) < 10 for s in ('validation', 'test') for k in ('chi', 'pon', 'none')):
        raise ValueError('Insufficient support for classwise holdout evaluation')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'features').mkdir()
    code = ['call_policy.py', 'scripts/train_call_repair.py', 'scripts/prepare_call_repair.py',
            'replay_call_repair.py', 'semantic.py', 'vendor/JevAny/jevany/model.py']
    protocol = {'config': cfg, 'code_sha256': {p: digest(ROOT / p) for p in code},
        'code_tree_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip(),
        'data_provenance_sha256': digest(data / 'provenance.json'),
        'reference_files': {p.name: digest(p) for p in reference.iterdir() if p.is_file()},
        'selection': cfg['selection'], 'test_used_for_selection': False}
    write(out / 'protocol.json', protocol)
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
    from jevany import JevModel
    from jevany.checkpoint import LoadOptions
    from jevany.inference import InferenceOptions
    from jevany.api import validate_distribution
    import os
    base = Path(os.environ.get('JEV_BASE_MODEL_DIR', Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
    if not base.is_dir(): raise ValueError('Pinned local base snapshot missing')
    write(out / 'status.json', {'status': 'loading_frozen_confirmed_model', 'step': 0})
    client = JevModel.from_pretrained(str(reference), device='cuda', dtype='bf16',
        options=LoadOptions(merge=False, temperature=1., compile_mode=None, cuda_graphs=False,
                            fused_kernels=False, base_load_path=str(base)),
        inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192,
                                          max_packed_tokens=8192, prefix_cache_size=0))
    model = client.runtime.model
    model.requires_grad_(False)
    frozen_head = model.head
    meta = torch.load(reference / 'head.pt', map_location='cpu', weights_only=True)
    initial = make_head(meta, 'cpu').eval().requires_grad_(False)
    capture = []
    hook = model.head.register_forward_pre_hook(lambda module, inputs: capture.append(tuple(t.detach() for t in inputs)))
    native_checks = 0
    def extract(split):
        nonlocal native_checks
        records = [json.loads(line) for line in (data / f'{split}.jsonl').read_text().splitlines()]
        rows = []
        for i, record in enumerate(records):
            if not is_call_request(record) or request_hash(record) != record['_meta']['text_sha256']:
                raise ValueError('Reaction or hash audit failed')
            question = record['questions']['action']; keys = list(question['criteria'])
            label = keys.index(question['label']); kinds = [v['type'] for v in question['criteria'].values()]
            for view in cfg['views']:
                request = payload(record, view)
                capture.clear()
                result = client.system_one(**request)
                if len(capture) != 1: raise ValueError('Expected one native pointer feature capture')
                decide, options = (t.clone().to('cpu', dtype=torch.float32) for t in capture.pop())
                if torch.is_inference(decide) or torch.is_inference(options): raise ValueError('Cache must permit query autograd')
                probabilities, _ = validate_distribution(result['answers']['action']['probabilities'], keys)
                with torch.no_grad():
                    cached = initial(decide, options).softmax(-1)
                if len(options) != len(keys) or not torch.allclose(torch.tensor(probabilities), cached, atol=2e-5, rtol=2e-4):
                    raise ValueError('Cached features do not reproduce native probabilities')
                native_checks += 1
                rows.append({'id': record['_meta']['id'], 'group': record['_meta']['group_id'],
                    'view': view, 'decide': decide, 'options': options, 'label': label, 'types': kinds})
            if (i + 1) % 64 == 0 or i + 1 == len(records):
                torch.save(rows[-64 * len(cfg['views']):], out / 'features' / f'{split}-{i // 64:04}.pt')
                write(out / 'status.json', {'status': 'encoding_repaired_reactions', 'split': split,
                    'raw_completed': i + 1, 'raw_total': len(records), 'views': len(cfg['views']), 'step': 0})
                print(json.dumps({'phase': 'encoding', 'split': split, 'raw_completed': i + 1, 'raw_total': len(records)}), flush=True)
        return rows
    train = extract('train')
    validation = extract('validation')
    head = make_head(meta, 'cuda').train()
    head.requires_grad_(False); head.q.requires_grad_(True)
    if any(p.requires_grad for p in model.parameters()): raise ValueError('Backbone or deployed head became trainable')
    optimizer = torch.optim.AdamW(head.q.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    baseline = evaluate(head, validation, 'cuda')
    write(out / 'validation-baseline.json', baseline)
    best, best_step, stale = baseline, 0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    curve = [{'step': 0, **baseline}]
    generator = random.Random(cfg['seed'])
    started = time.monotonic()
    for step in range(1, cfg['max_steps'] + 1):
        head.train()
        batch = generator.sample(train, min(cfg['batch_size'], len(train)))
        labels = torch.tensor([r['label'] for r in batch], device='cuda')
        loss = F.cross_entropy(logits_batch(head, batch, 'cuda'), labels)
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.q.parameters(), 1.)
        optimizer.step()
        if step % cfg['eval_every'] != 0: continue
        head.eval(); report = evaluate(head, validation, 'cuda')
        curve.append({'step': step, 'training_loss': float(loss.detach()), 'gradient_norm': float(norm), **report})
        checkpoint = out / 'checkpoints' / f'step-{step:06d}'
        checkpoint.mkdir(parents=True)
        state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if not all(torch.equal(state[k], meta['head'][k]) for k in ('k.weight', 'k.bias')):
            raise ValueError('Frozen key changed')
        torch.save({'head': state, 'reference_head_sha256': cfg['reference_head_sha256'],
            'scope': sorted(['chi', 'pon', 'none']), 'step': step, 'method': cfg['trainable']}, checkpoint / 'call-head.pt')
        torch.save({'optimizer': optimizer.state_dict(), 'step': step}, checkpoint / 'optimizer-private.pt')
        write(checkpoint / 'validation.json', report)
        better = (-report['macro_f1'], report['nll'], step) < (-best['macro_f1'], best['nll'], best_step)
        if better:
            best, best_step, best_state, stale = report, step, state, 0
        else: stale += 1
        write(out / 'validation-curve.json', curve)
        status = {'status': 'training_call_query', 'step': step, 'best_step': best_step,
            'validation_macro_f1': report['macro_f1'], 'baseline_macro_f1': baseline['macro_f1'],
            'validation_nll': report['nll'], 'stale_checks': stale, 'seconds': time.monotonic() - started}
        write(out / 'status.json', status); print(json.dumps(status), flush=True)
        if stale >= cfg['patience']: break
    selected = {'step': best_step, 'validation': best, 'baseline': baseline,
        'checkpoint': str((out / 'checkpoints' / f'step-{best_step:06d}' / 'call-head.pt').relative_to(ROOT)) if best_step else None,
        'selection': cfg['selection'], 'test_opened_at_selection': False}
    write(out / 'selected.json', selected)
    # Open test only after the checkpoint is frozen and cannot be changed by test outcomes.
    test = extract('test')
    head.load_state_dict(best_state); head.eval()
    testing = {'selected_step': best_step, 'candidate': evaluate(head, test, 'cuda'),
        'reference': evaluate(initial.to('cuda'), test, 'cuda'), 'used_for_selection': False}
    write(out / 'test-summary.json', testing)
    # Actual native routing, not just cached-head metrics, must reproduce selected logits.
    router = CallRouter(client, head)
    test_records = [json.loads(s) for s in (data / 'test.jsonl').read_text().splitlines()[:3]]
    native_rows = {r['id']: r for r in test if r['view'] == 'native'}
    for record in test_records:
        req = payload(record, 'native'); response = router.system_one(**req)
        keys = list(req['questions']['action']['criteria'])
        ordered, _ = validate_distribution(response['answers']['action']['probabilities'], keys)
        cached_row = native_rows[record['_meta']['id']]
        with torch.no_grad():
            expected = head(cached_row['decide'].cuda(), cached_row['options'].cuda()).softmax(-1).cpu()
        if not torch.allclose(torch.tensor(ordered), expected, atol=2e-5, rtol=2e-4):
            raise ValueError('Native call routing differs from selected cached head')
        if model.head is not frozen_head: raise ValueError('Router did not restore source')
    # Existing discard cases are used only for implementation identity, never to tune the call model.
    for line in (ROOT / 'data/semantic-2026-v1/validation.jsonl').read_text().splitlines():
        record = json.loads(line)
        kinds = {v['type'] for v in record['questions']['action']['criteria'].values()}
        if kinds != {'dahai'}: continue
        req = payload(record, 'native')
        before = client.system_one(**req)['answers']['action']['probabilities']
        after = router.system_one(**req)['answers']['action']['probabilities']
        if before != after or model.head is not frozen_head: raise ValueError('Noncall inference changed')
        break
    else: raise ValueError('No discard identity regression case')
    hook.remove()
    if any(digest(reference / name) != h for name, h in protocol['reference_files'].items()):
        raise ValueError('Frozen reference files changed')
    checks = {'native_feature_probability_checks': native_checks, 'native_router_checks': len(test_records),
        'frozen_key_equal': True, 'frozen_backbone_and_lora': True,
        'reference_files_unchanged': True, 'discard_head_training_updates': 0,
        'nonreaction_native_probabilities_identical': True}
    write(out / 'runtime-audit.json', checks)
    result = {'status': 'pilot_complete' if best_step else 'stop_no_validation_progress',
        'selected': selected, 'test': testing, 'audit': checks, 'deployed': False,
        'strength_evaluation_complete': False}
    write(out / 'training-result.json', result)
    write(out / 'status.json', {'status': result['status'], 'best_step': best_step,
        'strength_evaluation_complete': False, 'deployed': False})
    print(json.dumps({'phase': 'complete', 'best_step': best_step,
        'validation_macro_f1': best['macro_f1'], 'test_macro_f1': testing['candidate']['macro_f1']}), flush=True)


if __name__ == '__main__':
    main()
