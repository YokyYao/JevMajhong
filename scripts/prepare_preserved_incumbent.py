"""Freeze an untested positive incumbent for a separate, fresh follow-up study.

Historical training decisions remain unchanged. This study follows the user's
incremental-improvement rule and never selects another head on confirmation data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from head_rl import fingerprint,write
from incremental_score import ORIGINAL_HEAD,SELECTION_POLICY,choose_positive,confirmation_config,verified_development


def validate_frozen_runtime(candidate,benchmark):
    fields=['base','base_revision','lora','special_embeddings','backbone_adapter','multimodal',
            'head_dim','head_residual_dim','head_type','option_isolation','branch_mode',
            'decision_mode','verbalizers','temperature','weights_dtype','tokenizer_saved']
    for field in fields:
        if candidate.get(field)!=benchmark.get(field):
            raise ValueError(f'Frozen runtime metadata differs:{field}')


def study_config(source_config,seed_start,seeds):
    if seeds<128 or seed_start<0 or seed_start+seeds>2**32:
        raise ValueError('Follow-up requires at least128 valid independent logical streams')
    config={**source_config,'selection_policy':SELECTION_POLICY,
            'confirmation_qualification':'positive_paired_ci95','confirmation_min_ci95_lower':0,
            'confirmation_seed_start':seed_start,'confirmation_seeds':seeds,
            'desired_net_points_gain':500,'fixed_gain_gate_enabled':False}
    return confirmation_config(config)


def prepare(source,out,seed_start,seeds):
    if out.exists():raise ValueError('Preserve the existing frozen study')
    original,baseline,results=verified_development(source)
    chosen=choose_positive(results,baseline['mean_net_points'])
    if chosen is None:raise ValueError('The completed source has no positive incumbent')
    config=study_config(original['config'],seed_start,seeds)
    proposed=set(range(seed_start,seed_start+seeds))
    for path in (ROOT/'runs').glob('*/protocol.json'):
        previous=json.loads(path.read_text()).get('config',{})
        for start,length in [('training_seed_start','training_seeds_per_stage'),
                             ('development_seed_start','development_seeds'),
                             ('confirmation_seed_start','confirmation_seeds')]:
            if start in previous and length in previous:
                count=previous.get('max_stages',1) if start=='training_seed_start' else 1
                if proposed&set(range(previous[start],previous[start]+previous[length]*count)):
                    raise ValueError(f'Fresh confirmation range was already declared in {path}')
    checkpoint,benchmark=ROOT/chosen['checkpoint'],ROOT/config['reference']
    if fingerprint(benchmark/'head.pt')!=ORIGINAL_HEAD:raise ValueError('Original2000-step benchmark changed')
    validate_frozen_runtime(torch.load(checkpoint/'head.pt',map_location='cpu',weights_only=False),
                            torch.load(benchmark/'head.pt',map_location='cpu',weights_only=False))
    for name in ['adapter_config.json','adapter_model.safetensors','tokenizer.json','tokenizer_config.json','chat_template.jinja']:
        if fingerprint(checkpoint/name)!=fingerprint(benchmark/name):
            raise ValueError(f'Frozen input differs from original benchmark:{name}')
    head_sha=fingerprint(checkpoint/'head.pt');config['out']=str(out)
    selected={'checkpoint':chosen['checkpoint'],'candidate_head_sha256':head_sha,
              'development_update':chosen['update'],'config':config,
              'selection_rule':'preserve highest complete positive development incumbent; earliest tie',
              'source_run':str(source),'source_training_reference':original['config']['reference']}
    frozen={**original,'config':config,'sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
            'code_tree_sha':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),
            'reference_head_sha256':ORIGINAL_HEAD,
            'reference_adapter_sha256':fingerprint(benchmark/'adapter_model.safetensors'),
            'source_run':str(source),'source_training_reference':original['config']['reference'],
            'source_protocol_sha256':original['sha256'],
            'study_type':'fresh_single_preserved_positive_incumbent_followup',
            'historical_training_decisions_unchanged':True,
            'rule_source':'user incremental continuation clarification2026-10-30'}
    frozen['code_sha256']={**original['code_sha256'],'scripts/prepare_preserved_incumbent.py':fingerprint(Path(__file__)),
                          'incremental_score.py':fingerprint(ROOT/'incremental_score.py')}
    provenance=sorted(source.glob('training-games-stage-*.json'))
    if not provenance:raise ValueError('Missing original training provenance')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent,prefix=out.name+'.prepare-') as temporary:
        staged=Path(temporary)/'study';staged.mkdir()
        for name,data in [('protocol.json',frozen),('confirmation-selection.json',selected),
                          ('status.json',{'status':'awaiting_confirmation','checkpoint':chosen['checkpoint']})]:write(staged/name,data)
        for p in provenance+sorted(source.glob('reference-controls-stage-*.json')):shutil.copy2(p,staged/p.name)
        shutil.copy2(source/'protocol.json',staged/'source-training-protocol.json')
        shutil.copy2(source/'pilot-analysis.json',staged/'source-training-analysis.json')
        staged.rename(out)
    return {'out':str(out),'checkpoint':chosen['checkpoint'],'candidate_head_sha256':head_sha,
            'original_benchmark_head_sha256':ORIGINAL_HEAD,'fresh_seed_start':seed_start,'independent_streams':seeds,
            'qualification':'candidate mean>=0 and seed-clustered paired95CI lower>0;500 desired only'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed-start',type=int,required=True);parser.add_argument('--seeds',type=int,default=128)
    args=parser.parse_args();print(json.dumps(prepare(args.source,args.out,args.seed_start,args.seeds)),flush=True)
