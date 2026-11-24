"""Finite actual-model forward/backward probe, without an optimizer or fit.

Uses saved1200 only as an integration reference, not as a selected future parent.
Run inside the current allocation only with ample observed free memory; cap this
process's PyTorch allocator and preserve all original tensors/checkpoint files.
"""
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_ri_replay_epoch_adapter_v1 import EpochReplay,random_proxy
from tools import curriculum_ri_inverse_repeat_loss_v1 as weighted
from tools import curriculum_progression_parent_loss_v1 as retained
from tools.curriculum_ri_every_update_replay_v1 import RI


def main():
    allocation=resource();started=time.monotonic()
    memory=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.total,memory.free',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
    lines=memory.stdout.strip().splitlines()
    if len(lines)!=1:raise ValueError('Probe requires one physically allocated GPU on this node')
    uuid,total,free=[s.strip() for s in lines[0].split(',')]
    if int(free)<20000:raise ValueError('Insufficient observed free memory; do not load a companion model')
    parent=ROOT/'runs/curriculum-progression-parent-retention-v1'
    regpath=parent/'fit-plan/registered.json';reg=read(regpath)
    prerequisite=ROOT/'runs/curriculum-ri-replay-epoch-adapter-v1/completion-binding-private.json'
    bound=read(prerequisite)
    if any(sha(p)!=h for p,h in bound['files'].items()):raise ValueError('Frozen replay prerequisites changed')
    out=ROOT/'runs/curriculum-ri-replay-model-probe-v1'
    if out.exists():raise ValueError('Never overwrite a model probe attempt')
    out.mkdir()
    files=dict(bound['files']);files[str(prerequisite)]=sha(prerequisite)
    for p in (Path(__file__).resolve(),regpath):files[str(p)]=sha(p)
    sys.path[:0]=[reg['source'],reg['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.20,0)
    torch.manual_seed(2026112504)
    from peft import get_peft_model_state_dict
    from jevany import train as trainer
    from jevmahjong.policy.routing import load_model
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.initialization import exact_loaded_tensors
    base=Path(reg['training_config'].get('base_load_path') or
        Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17')
    expected_base={p:h for p,h in reg['inputs'].items() if Path(p).parent.resolve()==base.resolve()}
    if len(expected_base)<5 or not any(Path(p).suffix=='.safetensors' for p in expected_base):
        raise ValueError('Registered fixed base files required')
    if any(sha(p)!=h for p,h in expected_base.items()):raise ValueError('Fixed base file changed')
    files.update(expected_base);os.environ['JEV_BASE_MODEL_DIR']=str(base)
    checkpoint=parent/'stage3-fit/training-run/checkpoints/step-001200'
    identity=bundle(checkpoint)
    if identity['identity_sha256']!='d86892fc74f59b4c31f9028d87eca889347cd3f9406feefbf0f02b91c6fc6733':
        raise ValueError('The exact previously inspected1200 reference is required')
    for name in identity['files']:files[str(checkpoint/name)]=sha(checkpoint/name)
    write(out/'registered-private.json',{'status':'real_model_no_optimizer_probe_bound_before_load',
        'files':files,'reference_checkpoint':identity,'candidate_parent_selected':False,
        'observed_memory':memory.stdout,'pytorch_memory_fraction_cap':.20,'allocation':allocation,
        'optimizer_created':False,'student_updates':0,'new_games':0})
    receipt=read(reg['training_config']['data_receipt']);path=Path(receipt['files']['train']['path'])
    with path.open() as stream:rows=[json.loads(line) for line in stream]
    planroot=ROOT/'runs/curriculum-ri-every-update-replay-v1'
    plans=[read(planroot/f'epoch-{epoch:02d}-private.json') for epoch in (1,2)]
    adapter=EpochReplay(rows,plans);pool,admission=adapter.admit(rows,read(parent/'stage3-fit/token-admission.json'))
    args=SimpleNamespace(seed=reg['curriculum']['seed'],batch=1,accum=73,rlcr=False,
        p_none=0.,p_none_distract=0.,p_distract=0.,p_none_pair=0.,
        max_state=4096,max_branch=8192,max_packed=8192)
    original=dict(random=trainer.random,materialize=trainer.materialize,loss=trainer.question_loss)
    client=None;model=None;pending=[];losses=[];tokens=[]
    try:
        client=load_model(checkpoint);model=client.runtime.model
        if client.runtime.checkpoint.meta.base_revision!='2fc06364715b967f1860aea9cf38778875588b17':
            raise ValueError('Loaded reference uses a different base revision')
        exact_before=exact_loaded_tensors(get_peft_model_state_dict(model.lm,save_embedding_layers=False),model.head.state_dict(),checkpoint)
        named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
        if (len(named)!=376 or sum(n.startswith('head.') for n,p in named)!=4
                or any(not (n.startswith('head.') or 'lora_' in n) for n,p in named)):
            raise ValueError('Only the expected372 LoRA/four head tensors may require gradients')
        parameters=[p for n,p in named]
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.lm.config.use_cache=False;model.train();model.zero_grad(set_to_none=True)
        trainer.materialize=lambda r:retained.annotate(original['materialize'](r),r)
        trainer.random,created=random_proxy(original['random'],args.seed,adapter)
        trainer.question_loss=lambda z,q,device:weighted.weighted_loss(z,q,device,pending)
        rng=trainer.random.Random(args.seed);order=list(range(len(pool)));rng.shuffle(order)
        autocast=torch.autocast('cuda',dtype=torch.bfloat16)
        for position in order[:73]:
            batch=adapter.encode_batch(trainer.encode_batch,model,client.runtime.tok,args,[pool[position]],0)
            loss,terms=trainer.batch_loss(model,args,batch,'cuda',autocast)
            if not torch.isfinite(loss):raise ValueError('Nonfinite actual model objective')
            (loss/73).backward();losses.append(float(loss.detach()));tokens.extend(v.tokens for v in batch)
            if len(losses)%10==0:print(json.dumps({'phase':'real_forward_backward','records':len(losses)}),flush=True)
        observation=weighted.summarize(pending)
        if len(created)!=1 or len(pending)!=73 or any(observation['source_counts'].get(s)!=1 for s in RI):
            raise ValueError('First actual73-record update must contain all four RI strata once')
        error=abs(observation['loss']-sum(losses)/73)
        if error>1e-5:raise ValueError('Real model optimized loss and journal observer disagree')
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ValueError('Every expected trainable tensor needs finite actual gradients')
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
        if not math.isfinite(norm) or norm<=0:raise ValueError('Finite nonzero complete backward pass required')
        peak=torch.cuda.max_memory_allocated();model.zero_grad(set_to_none=True)
        exact_after=exact_loaded_tensors(get_peft_model_state_dict(model.lm,save_embedding_layers=False),model.head.state_dict(),checkpoint)
        if any(p.grad is not None for p in parameters) or bundle(checkpoint)!=identity:
            raise ValueError('Probe left gradients or modified original checkpoint files')
        write(out/'optimizer-observation-private.json',observation)
        result={'status':'actual_weighted_replay_first_batch_model_forward_backward_passed_no_optimizer',
            'reference_checkpoint_identity':identity['identity_sha256'],'records':len(losses),
            'actual_tokens':sum(tokens),'maximum_record_tokens':max(tokens),
            'actual_source_count':len(observation['source_counts']),'all_four_RI_sources_once':True,
            'finite_gradient_tensors':len(parameters),'preclip_gradient_norm':norm,
            'actual_weighted_objective':observation['loss'],'actual_weighted_standard_CE':observation['ce'],
            'optimized_vs_observed_loss_error':error,'peak_this_process_allocated_bytes':peak,
            'exact_before':exact_before,'exact_after':exact_after,'checkpoint_files_unchanged':True,
            'all_gradients_cleared':True,'model_encoder_stubbed':False,'model_loaded':True,
            'optimizer_created':False,'student_updates':0,'new_games':0,'candidate_parent_selected':False,
            'full_fit_or_policy_improvement_not_proven':True,'seconds':time.monotonic()-started,
            'pytorch_memory_fraction_cap':.20,'allocation':allocation}
        write(out/'summary-private.json',result)
        for p in (out/'registered-private.json',out/'optimizer-observation-private.json',out/'summary-private.json'):files[str(p)]=sha(p)
        write(out/'completion-binding-private.json',{'files':files})
        print(json.dumps({k:v for k,v in result.items() if k!='allocation'}),flush=True)
    finally:
        trainer.random=original['random'];trainer.materialize=original['materialize'];trainer.question_loss=original['loss']
        if model is not None:model.zero_grad(set_to_none=True)
        client=None;model=None;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
