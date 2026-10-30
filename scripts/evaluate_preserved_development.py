"""Add one fixed preserved-policy dev reference without changing primary decisions."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from head_rl import HeadPolicy,fingerprint,write
from preserved_development import compare_preserved
from scripts.audit_diagonal_value import audit_development
from scripts.prepare_preserved_incumbent import validate_frozen_runtime
from scripts.train_diagonal_value import BASE_FILES,ORIGINAL_HEAD
from scripts.train_head_rl import evaluate

PRESERVED_HEAD='b1248fbc9c9bd1a768a9eaf5fdb5241c82c96ac50eaabf8e0ca8fed885bec1ea'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--anchor',type=Path,default=Path('runs/head-rl-diagonal-value-20261030/checkpoints/calibration-02'))
    args=parser.parse_args();source=ROOT/args.source;anchor=ROOT/args.anchor
    original_protocol=json.loads((source/'protocol.json').read_text());original_config=original_protocol['config']
    native=source/'native-development';out=native/'preserved-anchor'
    if out.exists():raise ValueError('Existing secondary evaluation; inspect it rather than resubmit')
    audited=audit_development(source)
    protected={name:fingerprint(native/name) for name in ['selection.json','development-results.json']}
    if fingerprint(anchor/'head.pt')!=PRESERVED_HEAD:raise ValueError('Preserved policy changed')
    original=ROOT/original_config['reference']
    if fingerprint(original/'head.pt')!=ORIGINAL_HEAD:raise ValueError('Original SFT benchmark changed')
    validate_frozen_runtime(torch.load(anchor/'head.pt',map_location='cpu',weights_only=False),
                            torch.load(original/'head.pt',map_location='cpu',weights_only=False))
    for name in BASE_FILES:
        if fingerprint(anchor/name)!=fingerprint(original/name):raise ValueError('Preserved backbone inputs differ')
    config={**original_config,'reference':str(anchor),'out':str(out)}
    out.mkdir()
    native_protocol=json.loads((native/'protocol.json').read_text())
    write(out/'protocol.json',{'config':config,'source_protocol_sha256':original_protocol['sha256'],
          'preserved_head_sha256':PRESERVED_HEAD,'original_head_sha256':ORIGINAL_HEAD,
          'scope':'supplementary development reference only; no primary qualification or candidate selection changes',
          'code_sha256':{**native_protocol['code_sha256'],
                         'preserved_development.py':fingerprint(ROOT/'preserved_development.py'),
                         'scripts/evaluate_preserved_development.py':fingerprint(Path(__file__))},
          'protected_main_result_sha256':protected})
    random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    policy=HeadPolicy(config);policy.head.requires_grad_(False)
    _,rows=evaluate(policy,out/'native-matches',config,time.monotonic()+60*60,frozen=True)
    if policy.checks<1:raise ValueError('Native probability/action parity was not exercised')
    load_rows=lambda path:[json.loads(line) for line in path.read_text().splitlines()]
    baseline=load_rows(native/'baseline'/'games.jsonl')
    entries=json.loads((native/'development-results.json').read_text())
    variants=[{'variant':entry['variant'],'checkpoint':entry['checkpoint'],
               'rows':load_rows(native/f"variant-{entry['variant']:02d}"/'games.jsonl')} for entry in entries]
    report=compare_preserved(baseline,rows,variants,original_config)
    for name,expected in protected.items():
        if fingerprint(native/name)!=expected:raise ValueError('Primary selection/results changed during secondary evaluation')
    if fingerprint(anchor/'head.pt')!=PRESERVED_HEAD:raise ValueError('Preserved source head modified')
    report.update(preserved_head_sha256=PRESERVED_HEAD,native_parity_checks=policy.checks,
                  main_selected_variant=audited['selected']['variant'] if audited['selected'] else None,
                  protected_main_result_sha256=protected)
    write(out/'comparison-result.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
