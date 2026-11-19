"""Check fresh-base LoRA/head feasibility and teacher data coverage.

Run inside a GPU Slurm allocation after building the teacher source dataset.
This is a one-step probe, not a trained candidate or a strength benchmark.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/training/mortal-teacher-v1.json')
    p.add_argument('--out',type=Path,help='Separate probe-output directory; defaults to the configured teacher run')
    a=p.parse_args()
    from tools.build_mortal_teacher import require_gpu
    require_gpu()
    c=json.loads(a.config.read_text())
    source_root=ROOT/c['out']
    out=a.out if a.out is not None else source_root
    if (out/'base-route-probe-private.json').exists():
        raise RuntimeError('Existing probe: inspect its receipt instead of overwriting it')
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,c['upstream_source'])
    from jevmahjong.endgame.collect import write
    counts={};coverage={};groups={}
    for split in ('train','validation'):
     rows=[]
     for path in sorted((source_root/'sources').glob('*.json')):
      source=json.loads(path.read_text())
      if source['split']==split: rows.extend(source['pool'])
     counts[split]=len(rows);groups[split]=len({r['_meta']['group_id'] for r in rows})
     stats={}
     for kind in ('reach','chi','pon','ankan','daiminkan','kakan'):
      available=[r for r in rows if any(v['type']==kind for v in r['questions']['action']['criteria'].values())]
      positive=sum(r['questions']['action']['criteria'][r['_meta']['teacher_action']]['type']==kind for r in available)
      stats[kind]=dict(opportunities=len(available),teacher_selects=positive,teacher_declines=len(available)-positive)
     coverage[split]=stats
    inventory=dict(status='existing_teacher_source_inventory_complete',counts=counts,groups=groups,coverage=coverage,
     notes='These are ALL stored admitted teacher menus, not only selected1024/256. Existing split unchanged; no old-v8 retention targets included.')
    write(out/'base-route-data-inventory-private.json',inventory)
    import torch
    from peft import get_peft_model_state_dict
    from jevany.model import DecisionModel,load_preprocessor
    from tools.train_mortal_teacher import read_rows,labelled_training_row
    from jevany.data import materialize
    torch.manual_seed(2026112001)
    tok=load_preprocessor(c['base_model'],multimodal=False,backbone_adapter='auto')
    model=DecisionModel(c['base_model'],tok,'cuda',lora=16,head_dim=256,head_residual_dim=0,
     option_isolation=False,special_embeddings=False,lora_targets='all',lora_dropout=.05,
     dtype=torch.bfloat16,multimodal=False,backbone_adapter='auto',branch_mode='auto',decision_mode='pointer')
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.lm.config.use_cache=False
    initial=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
    b_keys=[k for k in initial if 'lora_B' in k]
    assert b_keys and all(torch.count_nonzero(initial[k]).item()==0 for k in b_keys)
    head0={k:v.detach().cpu().clone() for k,v in model.head.state_dict().items()}
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    headids={id(p) for p in model.head.parameters()}
    allowed=all(id(p) in headids or 'lora_' in n or 'trainable_tokens' in n for n,p in named)
    assert allowed
    rows=[r for r in read_rows(source_root/'train.jsonl') if r['questions']['action']['src'].startswith('teacher_')]
    samples=[];seen=set()
    for r in rows:
     src=r['questions']['action']['src']
     if src not in seen: samples.append(r);seen.add(src)
    assert len(samples)==7
    optimizer=torch.optim.AdamW([p for _,p in named],lr=1e-4)
    optimizer.zero_grad(set_to_none=True);model.train();losses=[]
    for r in samples:
     e=model.encode(tok,materialize(labelled_training_row(r)),max_state=4096,max_branch=8192,strict=True)
     assert not e.get('state_truncated') and len(e['ids'])<=8192
     with torch.autocast('cuda',dtype=torch.bfloat16): z=model.forward(e)[0]
     q=r['questions']['action'];t=torch.tensor([q['target'][k] for k in q['criteria']],device=z.device,dtype=torch.float32)
     loss=-(t*z.float().log_softmax(-1)).sum()/len(samples)
     assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach())*len(samples))
    norm=torch.nn.utils.clip_grad_norm_([p for _,p in named],1.,error_if_nonfinite=True);optimizer.step()
    actual=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
    assert any(torch.count_nonzero(actual[k]).item()>0 for k in b_keys)
    assert any(not torch.equal(v.detach().cpu(),head0[k]) for k,v in model.head.state_dict().items())
    report=dict(status='cold_base_one_step_feasibility_passed',base=c['base_model'],
     no_v8_checkpoint_or_adapter_or_head_loaded=True,tokenizer_from_base=True,initial_lora_B_all_zero=True,
     head_new_random_initialization=True,actual_backward_optimizer_step=True,actual_lora_and_head_updates=True,
     samples=7,mean_probe_ce=sum(losses)/len(losses),gradient_norm=float(norm),
     trainable_parameters=sum(p.numel() for _,p in named),special_embeddings=model.special_embeddings,
     branch_mode=model.branch_mode,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
     inventory=inventory,not_a_trained_candidate=True,no_checkpoint_saved=True,raw_publication=False)
    write(out/'base-route-probe-private.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
