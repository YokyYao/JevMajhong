"""Freeze the predeclared positive diagonal variant for untouched final walls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from head_rl import fingerprint,write
from scripts.audit_score_confirmation import require
from scripts.audit_diagonal_value import audit_development,audit_fit
from scripts.train_diagonal_value import ORIGINAL_HEAD


def prepare(source,out):
    require(not out.exists(),'Preserve existing partial or completed final study')
    fit_audit=audit_fit(source);development_audit=audit_development(source)
    chosen=development_audit['selected'];require(chosen is not None,'No positive native development candidate')
    original=json.loads((source/'protocol.json').read_text());config={**original['config'],'out':str(out)}
    require(config['confirmation_seeds']>=128 and config['target_net_points_gain']==0
            and config['confirmation_min_ci95_lower']==0 and config['fixed_gain_gate_enabled'] is False,
            'Keep predeclared positive paired evidence criterion')
    training_files=sorted(source.glob('training-games-stage-cache-*.json'))
    training={row['seed'] for p in training_files for row in json.loads(p.read_text())}
    development=set(range(config['development_seed_start'],config['development_seed_start']+config['development_seeds']))
    confirmation=set(range(config['confirmation_seed_start'],config['confirmation_seed_start']+config['confirmation_seeds']))
    require(training and not training&development and not training&confirmation and not development&confirmation,
            'Missing or overlapping training/development/confirmation streams')
    for path in (ROOT/'runs').glob('*/protocol.json'):
        if path.resolve()==(source/'protocol.json').resolve():continue
        previous=json.loads(path.read_text()).get('config',{})
        for start,length in [('training_seed_start','training_seeds_per_stage'),('development_seed_start','development_seeds'),
                             ('confirmation_seed_start','confirmation_seeds')]:
            if start in previous and length in previous:
                count=previous.get('max_stages',1) if start=='training_seed_start' else 1
                require(not confirmation&set(range(previous[start],previous[start]+previous[length]*count)),
                        f'Final wall range already declared:{path}')
    native=json.loads((source/'native-development'/'protocol.json').read_text())
    protocol={**original,'config':config,'sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
              'code_tree_sha':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),
              'code_sha256':{**native['code_sha256'],
                'scripts/audit_diagonal_value.py':fingerprint(ROOT/'scripts/audit_diagonal_value.py'),
                'scripts/prepare_diagonal_confirmation.py':fingerprint(Path(__file__))},
              'source_run':str(source),'source_protocol_sha256':original['sha256'],
              'reference_head_sha256':ORIGINAL_HEAD,'study_type':'diagonal_value_predeclared_positive_native_variant'}
    selection={'checkpoint':chosen['checkpoint'],'candidate_head_sha256':chosen['head_sha256'],'config':config,
               'source_run':str(source),'development_variant':chosen['variant'],
               'selection_rule':'highest complete positive native development net gain; earliest predefined variant on tie'}
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent,prefix=out.name+'.prepare-') as temporary:
        staged=Path(temporary)/'study';staged.mkdir()
        for name,value in [('protocol.json',protocol),('confirmation-selection.json',selection),
                           ('source-fit-audit.json',fit_audit),('source-development-audit.json',development_audit),
                           ('status.json',{'status':'awaiting_confirmation'})]:write(staged/name,value)
        for path in training_files:shutil.copy2(path,staged/path.name)
        shutil.copy2(source/'protocol.json',staged/'source-training-protocol.json')
        shutil.copy2(source/'native-development'/'selection.json',staged/'source-development-selection.json')
        staged.rename(out)
    return selection


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.source,args.out)),flush=True)
