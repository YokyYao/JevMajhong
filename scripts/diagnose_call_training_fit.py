"""Compare parent/current fit on executed TRAIN and unchanged validation.

Read-only inference: no optimization, new selection, threshold changes or test
opening. All per-request outputs stay private under runs/.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_fit_diagnostic import reaction_records, summarize_fit
from call_policy import digest
from checkpoint_identity import completed_checkpoint_files
from feature_continuation import verify_preparation
from head_rl import write


def registration(cfg):
    from jevany.data import load_records
    data_source = json.loads((ROOT / cfg['data_source_config']).read_text())
    source = ROOT / data_source['out']
    prepared = json.loads((source / 'preparation.json').read_text())
    verify_preparation(ROOT, data_source, prepared)
    datasets = {split: reaction_records(load_records(ROOT / data_source['data'] / (split + '.jsonl')), split)
                for split in ('train', 'validation')}
    if any(len(datasets[s]) != cfg['expected_records'][s] for s in datasets):
        raise ValueError('Registered diagnostic coverage differs')
    if ({r['_meta']['group_id'] for r in datasets['train']} &
            {r['_meta']['group_id'] for r in datasets['validation']}):
        raise ValueError('Whole-match TRAIN/validation overlap')
    models, sources = {}, {}
    for name, config_path in cfg['models'].items():
        model_cfg = json.loads((ROOT / config_path).read_text())
        run = ROOT / model_cfg['out']
        preparation = json.loads((run / 'preparation.json').read_text())
        verify_preparation(ROOT, model_cfg, preparation)
        result = json.loads((run / 'result.json').read_text())
        if result['status'] != 'clean_policy_candidate' or result['selected_step'] <= 0:
            raise ValueError('Completed selected candidate required')
        models[name] = {'checkpoint': result['checkpoint'], 'step': result['selected_step'],
                        'files': completed_checkpoint_files(ROOT / result['checkpoint'], result['selected_step']),
                        'validation_rows': str(run.relative_to(ROOT) / 'training-run/training_eval' /
                                               f"step-{result['selected_step']:06d}/development/rows.json")}
        for path in (ROOT / config_path, run / 'result.json', run / 'preparation.json',
                     ROOT / models[name]['validation_rows']):
            sources[str(path.relative_to(ROOT))] = digest(path)
    for path in (ROOT / cfg['data_source_config'], source / 'preparation.json',
                 *(ROOT / data_source['data'] / (s + '.jsonl') for s in datasets)):
        sources[str(path.relative_to(ROOT))] = digest(path)
    code = {n: digest(ROOT / n) for n in ('call_fit_diagnostic.py',
            'scripts/diagnose_call_training_fit.py', 'call_training_fit.slurm',
            'targeted_reaction_sampling.py', 'vendor/JevAny/jevany/data.py',
            'vendor/JevAny/jevany/benchmark.py', 'vendor/JevAny/jevany/inference.py',
            'vendor/JevAny/jevany/model.py', 'vendor/JevAny/jevany/checkpoint.py',
            'vendor/JevAny/jevany/backbones.py')}
    return {'config': cfg, 'models': models, 'sources': sources, 'code': code,
            'optimizer_updates': 0, 'human_test_opened': False, 'deployed': False}, datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    protocol, datasets = registration(cfg)
    out = ROOT / cfg['out']
    if args.prepare:
        out.mkdir(parents=True, exist_ok=False)
        write(out / 'protocol.json', protocol)
        write(out / 'status.json', {'status': 'prepared_read_only_fit_diagnostic',
                                   'records': {s: len(r) for s, r in datasets.items()}})
        print(json.dumps({'prepared': True, 'records': cfg['expected_records']}))
        return
    if json.loads((out / 'protocol.json').read_text()) != protocol or (out / 'summary.json').exists():
        raise ValueError('Changed or already-completed fit diagnostic')
    import torch
    from jevany import JevModel
    from jevany.benchmark import prediction_rows
    from jevany.checkpoint import LoadOptions
    from jevany.data import api_request
    from jevany.inference import InferenceOptions
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    base = Path(os.environ.get('JEV_BASE_MODEL_DIR', Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
    summaries = {}
    started = time.monotonic()
    for name, identity in protocol['models'].items():
        model = JevModel.from_pretrained(str(ROOT / identity['checkpoint']), device='cuda', dtype='bf16',
            options=LoadOptions(merge=False, temperature=1., compile_mode=None, cuda_graphs=False,
                                fused_kernels=False, base_load_path=str(base)),
            inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192,
                                               max_packed_tokens=8192, prefix_cache_size=0))
        summaries[name] = {}
        expected_validation = {r['id']: r for r in json.loads((ROOT / identity['validation_rows']).read_text())}
        maximum_error = 0.
        for split, records in datasets.items():
            rows = []
            with (out / (name + '-' + split + '-private.jsonl')).open('x') as stream:
                for i, record in enumerate(records, 1):
                    answer = model.system_one(**api_request(record))['answers']['action']['probabilities']
                    row, = prediction_rows(record, {'probabilities': {'action': answer}})
                    if split == 'validation':
                        expected = expected_validation[row['id']]
                        if row['keys'] != expected['keys'] or row['label'] != expected['label']:
                            raise ValueError('Validation parity options/label differ')
                        error = max(abs(a - b) for a, b in zip(row['p'], expected['p']))
                        maximum_error = max(maximum_error, error)
                        if error > cfg['parity_atol'] or max(range(len(row['p'])), key=row['p'].__getitem__) != max(range(len(expected['p'])), key=expected['p'].__getitem__):
                            raise ValueError('Saved validation native parity differs')
                    stream.write(json.dumps(row) + '\n')
                    rows.append(row)
                    if i % 100 == 0 or i == len(records):
                        stream.flush()
                        write(out / 'status.json', {'status': 'fit_diagnostic', 'model': name,
                            'split': split, 'records': i, 'total_records': len(records),
                            'seconds': time.monotonic() - started})
            summaries[name][split] = summarize_fit(records, rows)
        summaries[name]['validation_parity'] = {'checked': len(datasets['validation']),
                                                'maximum_probability_error': maximum_error}
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    if registration(cfg)[0] != protocol:
        raise ValueError('Diagnostic inputs changed during inference')
    result = {'status': 'passed_complete', 'summaries': summaries,
              'records': cfg['expected_records'], 'optimizer_updates': 0,
              'not_strength_evidence': True, 'human_test_opened': False, 'deployed': False}
    write(out / 'summary.json', result)
    write(out / 'status.json', {'status': 'passed_complete', 'records': cfg['expected_records'],
                              'optimizer_updates': 0})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
