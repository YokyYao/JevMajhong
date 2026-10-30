"""Complete fresh East development games for predefined diagonal heads."""
import argparse
import copy
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from head_rl import HeadPolicy,fingerprint,paired_primary_comparison,write
from scripts.prepare_preserved_incumbent import validate_frozen_runtime
from scripts.train_diagonal_value import BASE_FILES,ORIGINAL_HEAD
from scripts.train_head_rl import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    source=ROOT/args.run;protocol=json.loads((source/'protocol.json').read_text());config=protocol['config']
    fit=json.loads((source/'training-result.json').read_text())
    if fit['status']!='fit_complete' or not fit['candidates']:raise ValueError('No predeclared native candidates')
    for name,sha in protocol['code_sha256'].items():
        if fingerprint(ROOT/name)!=sha:raise ValueError(f'Frozen fitter changed:{name}')
    reference=ROOT/config['reference']
    if fingerprint(reference/'head.pt')!=ORIGINAL_HEAD:raise ValueError('Original benchmark changed')
    meta=torch.load(reference/'head.pt',map_location='cpu',weights_only=False)
    proposed=set(range(config['development_seed_start'],config['development_seed_start']+config['development_seeds']))
    for path in (ROOT/'runs').glob('*/protocol.json'):
        if path.resolve()==(source/'protocol.json').resolve():continue
        other=json.loads(path.read_text()).get('config',{})
        for start,length in [('training_seed_start','training_seeds_per_stage'),('development_seed_start','development_seeds'),
                             ('confirmation_seed_start','confirmation_seeds')]:
            if start in other and length in other:
                count=other.get('max_stages',1) if start=='training_seed_start' else 1
                if proposed&set(range(other[start],other[start]+other[length]*count)):
                    raise ValueError(f'Development wall range already declared:{path}')
    # Verify all predefined variants before exposing any development outcomes.
    for item in fit['candidates']:
        checkpoint=ROOT/item['checkpoint']
        if fingerprint(checkpoint/'head.pt')!=item['head_sha256']:raise ValueError('Fitted head changed')
        validate_frozen_runtime(torch.load(checkpoint/'head.pt',map_location='cpu',weights_only=False),meta)
        for name in BASE_FILES:
            if fingerprint(reference/name)!=fingerprint(checkpoint/name):raise ValueError('Frozen model inputs differ')
    out=source/'native-development'
    if out.exists():raise ValueError('Preserve partial or completed development; do not resubmit')
    out.mkdir()
    write(out/'protocol.json',{'source_protocol_sha256':protocol['sha256'],'config':config,
          'preselected_candidates':fit['candidates'],'selection':'highest complete positive development net gain; earliest tie',
          'code_tree_sha':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),
          'code_sha256':{**protocol['code_sha256'],str(Path(__file__).relative_to(ROOT)):fingerprint(Path(__file__))}})
    random.seed(config['seed']);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    deadline=time.monotonic()+60*config['development_wall_minutes']
    policy=HeadPolicy(config);policy.head.requires_grad_(False)
    baseline,baseline_rows=evaluate(policy,out/'baseline',config,deadline,frozen=True)
    write(out/'baseline.json',baseline);results=[]
    for index,item in enumerate(fit['candidates'],1):
        write(out/'status.json',{'status':'evaluating_candidate','variant':index,'checkpoint':item['checkpoint']})
        metadata=torch.load(ROOT/item['checkpoint']/'head.pt',map_location='cpu',weights_only=False)
        policy.head.load_state_dict(metadata['head']);policy.head.eval();policy.checks=0
        report,rows=evaluate(policy,out/f'variant-{index:02d}',config,deadline)
        if policy.checks<1:raise ValueError('Native parity was not exercised')
        paired=paired_primary_comparison(rows,baseline_rows,config)
        result={'variant':index,**item,'metrics':report,'paired_exploratory':paired,'native_parity_checks':policy.checks}
        results.append(result);write(out/'development-results.json',results)
        print('complete_variant '+json.dumps(result),flush=True)
    positive=[r for r in results if r['paired_exploratory']['primary_delta']>0]
    selected=max(positive,key=lambda r:(r['metrics']['mean_net_points'],-r['variant'])) if positive else None
    if fingerprint(reference/'head.pt')!=ORIGINAL_HEAD:raise ValueError('Source benchmark modified')
    for item in fit['candidates']:
        if fingerprint(ROOT/item['checkpoint']/'head.pt')!=item['head_sha256']:raise ValueError('Candidate modified')
    result={'status':'complete','selected':selected,'baseline':baseline,'results':results,
            'fixed_gain_gate_enabled':False,'independent_improvement_confirmed':False,
            'note':'All predefined variants tested once. Native development selects a candidate; it does not establish improvement.'}
    write(out/'selection.json',result);write(out/'status.json',{'status':'complete','selected_variant':selected['variant'] if selected else None})
    print('development_selection '+json.dumps(result),flush=True)


if __name__=='__main__':main()
