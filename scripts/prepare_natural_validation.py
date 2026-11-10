"""Prepare fixed-model validation diagnostics only after complete native cache audit."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    args = p.parse_args(); cfg = json.loads(args.config.read_text())
    cache, data, out = (ROOT / cfg[k] for k in ('cache_source', 'data', 'out'))
    protocol = json.loads((cache / 'protocol.json').read_text())
    audit = json.loads((cache / 'runtime-audit.json').read_text())
    provenance = json.loads((data / 'provenance.json').read_text())
    expected = sum(provenance['counts'][s] for s in ('train', 'validation')) * len(cfg['views'])
    if (audit['status'] != 'feature_cache_complete' or audit['feature_splits'] != ['train', 'validation']
        or audit['test_opened'] is not False or audit['parameter_updates'] != 0
        or not audit['reference_files_unchanged'] or audit['native_feature_probability_checks'] != expected):
        raise ValueError('Complete audited train/validation cache required; no test or partial cache')
    if digest(data / 'provenance.json') != protocol['data_provenance_sha256']: raise ValueError('Data identity mismatch')
    if protocol['config']['data'] != cfg['data'] or protocol['config']['reference_head_sha256'] != cfg['reference_head_sha256']:
        raise ValueError('Cache source model/data mismatch')
    tensors = audit['tensor_sha256']
    if {p.name for p in (cache / 'features').glob('*.pt')} != set(tensors): raise ValueError('Missing or extra feature shards')
    for name, h in tensors.items():
        if digest(cache / 'features' / name) != h: raise ValueError('Cached tensor changed')
    for name, h in protocol['reference_files'].items():
        if digest(ROOT / cfg['reference'] / name) != h: raise ValueError('Frozen reference changed')
    source = ROOT / cfg['selection_origin'] / 'selected.json'
    selection = json.loads(source.read_text())
    if digest(ROOT / selection['checkpoint']) != cfg['expected_checkpoint_sha256']['candidate']:
        raise ValueError('Cannot change the frozen candidate')
    if digest(ROOT / cfg['previous_call_checkpoint']) != cfg['expected_checkpoint_sha256']['previous']:
        raise ValueError('Cannot change the previous comparison model')
    out.mkdir(parents=True, exist_ok=False); shutil.copyfile(source, out / 'selected.json')
    manifest = {'status': 'fixed_validation_diagnostic_prepared', 'test_opened': False,
        'parameter_updates': 0, 'model_selection_performed': False,
        'source_selected_sha256': digest(source), 'cache_audit_sha256': digest(cache / 'runtime-audit.json'),
        'data_provenance_sha256': digest(data / 'provenance.json'),
        'validation_records': provenance['counts']['validation'], 'validation_matches': len(provenance['matches']['validation']),
        'note': 'Copied selected.json retains historical selection statistics; new validation results are diagnostic only.'}
    (out / 'diagnostic-protocol.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__': main()
