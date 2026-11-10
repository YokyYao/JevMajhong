"""Freeze one clean policy, verify serving parity, then compare full-East points."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from call_policy import digest
from clean_policy_runtime import CleanPolicyRouter
from clean_policy_evaluation import aggregate, POLICIES, SUITES
from head_rl import write
from scripts.evaluate_call_repair import Agent
from scripts.evaluate_guide_calls import game

CODE = ['scripts/evaluate_clean_policy.py','scripts/audit_clean_policy.py','clean_policy_evaluation.py',
        'clean_policy_runtime.py','clean_policy_data.py','scripts/evaluate_guide_calls.py',
        'scripts/evaluate_call_repair.py','semantic.py','arena.py','scoring_audit.py','seed_control.py',
        'guide_development_audit.py','scripts/audit_guide_calls.py',
        'vendor/JevAny/jevany/inference.py','vendor/JevAny/jevany/data.py','vendor/JevAny/jevany/client.py']


def native_parity(router, run, preparation, selected_step, count_per_phase, atol):
    """Use already-opened validation only; preserve options and strip all labels."""
    from jevany.data import api_request
    path = ROOT/preparation['config']['data']/'validation.jsonl'
    records = [json.loads(l) for l in path.read_text().splitlines()]
    rows = json.loads((run/'training-run'/'training_eval'/f'step-{selected_step:06d}'/'development'/'rows.json').read_text())
    index = {r['id']:r for r in rows}
    counts=Counter(); max_error=0.; checked=0
    for record in sorted(records,key=lambda r:r['_meta']['id']):
        phase = record['_meta']['phase']; view=record['_meta']['view']; key=(phase,view)
        if counts[key]>=count_per_phase:
            continue
        expected = index[record['_meta']['id']]
        keys=list(record['questions']['action']['criteria'])
        if keys!=expected['keys']:
            raise ValueError('Parity option order changed')
        result=router.system_one(**api_request(record))['answers']['action']['probabilities']
        if list(result)!=keys:
            raise ValueError('Serving option keys/order changed')
        actual=torch.tensor([result[k] for k in keys],dtype=torch.float64)
        target=torch.tensor(expected['p'],dtype=torch.float64)
        if not torch.isfinite(actual).all() or not torch.isclose(actual.sum(),torch.tensor(1.,dtype=torch.float64),atol=1e-6):
            raise ValueError('Invalid serving probabilities')
        error=float((actual-target).abs().max());max_error=max(max_error,error)
        if error>atol or actual.argmax()!=target.argmax():
            raise ValueError(f'Training/serving parity failed: {phase}/{view}, error={error}')
        counts[key]+=1;checked+=1
    if len(counts)!=9 or min(counts.values())!=count_per_phase:
        raise ValueError('Missing phase/view serving parity coverage')
    return {'checked':checked,'counts':{p+'/'+v:n for (p,v),n in counts.items()},'max_absolute_probability_error':max_error,
        'atol':atol,'decisions_equal':True,'source':'already-opened training validation; no test or fresh development games'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args();cfg=json.loads(a.config.read_text())
    out=ROOT/cfg['out'];out.mkdir(parents=True,exist_ok=a.resume)
    source_cfg=json.loads((ROOT/cfg['source_config']).read_text());run=ROOT/source_cfg['out']
    result=json.loads((run/'result.json').read_text())
    if result['status']!='clean_policy_candidate' or result['selected_step']<=0:
        write(out/'evaluation-skipped.json',{'reason':result['status'],'deployed':False});return
    source=json.loads((run/'protocol.json').read_text());reference=ROOT/source_cfg['reference']
    if source['config']!=source_cfg:raise ValueError('Training config changed')
    for name,h in source['code_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Training code changed before evaluation')
    for name,h in source['reference_files'].items():
        if digest(reference/name)!=h:raise ValueError('FrozenB changed')
    checkpoint=ROOT/result['checkpoint']
    candidate_files={p.name:digest(p) for p in checkpoint.iterdir() if p.is_file() and p.name not in ('trainer_state.pt',)}
    if not all(name in candidate_files for name in ('adapter_config.json','adapter_model.safetensors','head.pt','complete.json')):
        raise ValueError('Selected candidate checkpoint incomplete')
    expected_start=source_cfg['development_seed_start'] if cfg['stage']=='development' else source_cfg['confirmation_seed_start']
    expected_n=source_cfg['development_seeds'] if cfg['stage']=='development' else source_cfg['confirmation_seeds']
    if cfg['stage'] not in ('development','independent_confirmation') or cfg['seed_start']!=expected_start or cfg['seeds']!=expected_n:
        raise ValueError('Unregistered strength evaluation walls/stage')
    if cfg['stage']=='independent_confirmation':
        development=ROOT/cfg['development_out']
        dev_audit=json.loads((development/'evaluation-audit.json').read_text())
        dev_protocol=json.loads((development/'protocol.json').read_text())
        if dev_audit['status']!='passed_complete' or dev_audit['overall_paired_gain']['gain']<=0:
            raise ValueError('Positive completed audited development signal required')
        if dev_protocol['candidate_files']!=candidate_files or dev_protocol['training_result_sha256']!=digest(run/'result.json'):
            raise ValueError('Confirmation must freeze the exact development candidate')
    protocol={'config':cfg,'training_config':source_cfg,'training_result_sha256':digest(run/'result.json'),
        'candidate_checkpoint':result['checkpoint'],'candidate_files':candidate_files,
        'reference_files':source['reference_files'],'code_sha256':{n:digest(ROOT/n) for n in CODE},
        'selected_step':result['selected_step'],'candidate_frozen':True,'human_test_opened':False,
        'opponents':{'closed':'3 EfficiencyAgent','mixed':'2 EfficiencyAgent +CallingOpponent at focal+1',
                     'strong':'2 EfficiencyAgent +frozenB at focal+1'},
        'guard':'same guarded discard set for both; retain legal reach; auto-win and oldB rare-action fallback',
        'primary':'paired complete-East terminal net points; four seats and three suites clustered within whole wall',
        'seed_scheme':'spaced-v1','deployed':False}
    target=out/'protocol.json'
    if target.exists() and json.loads(target.read_text())!=protocol:
        raise ValueError('Resume protocol/source/weights differ')
    if not target.exists():write(target,protocol)
    preparation=json.loads((run/'preparation.json').read_text())
    if digest(ROOT/source_cfg['data']/'validation.jsonl')!=preparation['files']['validation']:
        raise ValueError('Parity validation changed')
    from jevany import JevModel
    from jevany.checkpoint import LoadOptions
    from jevany.inference import InferenceOptions
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    base=Path(os.environ.get('JEV_BASE_MODEL_DIR',Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
    def load(path):
        return JevModel.from_pretrained(str(path),device='cuda',dtype='bf16',
            options=LoadOptions(merge=False,temperature=1.,compile_mode=None,cuda_graphs=False,fused_kernels=False,base_load_path=str(base)),
            inference_options=InferenceOptions(max_state_tokens=4096,max_branch_tokens=8192,max_packed_tokens=8192,prefix_cache_size=0))
    write(out/'status.json',{'status':'loading_frozen_clean_and_reference','games':0})
    ordinary=load(reference);candidate=load(checkpoint);router=CleanPolicyRouter(ordinary,candidate)
    parity=native_parity(router,run,preparation,result['selected_step'],cfg['parity_per_phase_view'],cfg['parity_atol'])
    write(out/'native-parity.json',parity)
    agents={'reference':Agent(ordinary),'candidate':Agent(router)};strong=Agent(ordinary)
    path=out/'games-private.jsonl'
    rows=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    completed={(r['suite'],r['seed'],r['seat'],r['policy']) for r in rows}
    if len(completed)!=len(rows):raise ValueError('Duplicate resumed game')
    expected={(s,n,p,k) for s in SUITES for n in range(cfg['seed_start'],cfg['seed_start']+cfg['seeds']) for p in range(4) for k in POLICIES}
    if not completed<=expected:raise ValueError('Foreign resumed game')
    started=time.monotonic()
    with path.open('a') as stream:
        for suite in SUITES:
            for seed in range(cfg['seed_start'],cfg['seed_start']+cfg['seeds']):
                for seat in range(4):
                    for name in POLICIES:
                        if (suite,seed,seat,name) in completed:continue
                        row={'policy':name,**game(agents[name],strong,seed,seat,suite)}
                        stream.write(json.dumps(row)+'\n');stream.flush();rows.append(row)
                        status={'status':'clean_policy_strength','stage':cfg['stage'],'suite':suite,
                            'games':len(rows),'total_games':len(expected),'seconds':time.monotonic()-started}
                        write(out/'status.json',status);print(json.dumps(status),flush=True)
    write(out/'summary.json',aggregate(rows,cfg))
    write(out/'status.json',{'status':'games_complete_audit_pending','games':len(rows),'deployed':False})


if __name__=='__main__':main()
