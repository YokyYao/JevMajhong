"""Loss and validation arithmetic for unified fresh-base Mahjong adaptation."""
import math
import torch


def question_loss(logits,question,device):
    """Do not push down legal actions that have no counterfactual outcome label."""
    if question.get('target') is not None:
        target=torch.tensor(question['target'],dtype=logits.dtype,device=device)
        if target.shape!=logits.shape or not torch.isfinite(target).all() or (target<0).any() or target.sum()<=0:
            raise ValueError('Valid complete target vector required')
        if question.get('src')=='paired_future_masked':
            mask=target>0;target=target[mask];logits=logits[mask]
        target=target/target.sum()
        return -(target*logits.log_softmax(-1)).sum()
    label=torch.tensor([question['label']],device=device)
    return torch.nn.functional.cross_entropy(logits[None],label)


def target_cross_entropy(question,keys,probabilities):
    """Recompute the same masked CE from actual returned probabilities."""
    if len(keys)!=len(probabilities) or len(set(keys))!=len(keys):raise ValueError('Prediction menu differs')
    values=dict(zip(keys,probabilities))
    if set(values)!=set(question['criteria']) or any(not math.isfinite(p) or p<0 for p in probabilities):
        raise ValueError('Invalid actual prediction distribution')
    target=question.get('target')
    if target is None:return -math.log(max(values[question['label']],1e-30))
    selected={key:value for key,value in target.items() if value>0}
    mass=sum(values[key] for key in selected) if question.get('src')=='paired_future_masked' else sum(probabilities)
    if mass<=0:raise ValueError('No predicted mass on labelled support')
    total=sum(selected.values())
    return -sum(value/total*math.log(max(values[key]/mass,1e-30)) for key,value in selected.items())


def validation_profile(records,rows,weights):
    by_id={r['_meta']['id']:r for r in records}
    if len(by_id)!=len(records) or len(rows)!=len(records) or {r['id'] for r in rows}!=set(by_id):
        raise ValueError('Validation dropped, repeated or changed records')
    totals={};counts={};gold={}
    for row in rows:
        record=by_id[row['id']];q=record['questions']['action'];phase=record['_meta']['phase']
        value=target_cross_entropy(q,row['keys'],row['p'])
        totals[phase]=totals.get(phase,0.)+value;counts[phase]=counts.get(phase,0)+1
        if record['_meta']['label_type']=='expert_replay':
            predicted=row['keys'][max(range(len(row['p'])),key=row['p'].__getitem__)]
            pair=gold.setdefault(phase,[0,0]);pair[0]+=predicted==q['label'];pair[1]+=1
    if set(totals)!=set(weights) or any(not math.isfinite(w) or w<=0 for w in weights.values()):
        raise ValueError('Registered validation phase weights or coverage differ')
    means={p:totals[p]/counts[p] for p in totals}
    return dict(weighted_target_ce=sum(weights[p]*means[p] for p in weights)/sum(weights.values()),
        phase_target_ce=means,phase_records=counts,
        expert_accuracy={p:a/n for p,(a,n) in gold.items()},
        pseudo_labels_are_not_expert_truth=True)


def package_digest(package):
    import hashlib
    result=hashlib.sha256()
    for path in sorted(package.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.pyc':continue
        result.update(path.relative_to(package).as_posix().encode());result.update(b'\0')
        result.update(hashlib.sha256(path.read_bytes()).digest())
    return result.hexdigest()


def train(cfg):
    """Use the frozen upstream trainer with an explicit masked-loss adapter."""
    import copy
    import json
    from pathlib import Path
    import sys
    import time
    import tomllib
    from jevmahjong.core.io import digest,write
    from jevmahjong.data.base_lora_pool import INSTRUCTION
    read=lambda p:json.loads(Path(p).read_text())
    upstream=Path(cfg['upstream_source']).resolve()
    if package_digest(upstream/'jevany')!=cfg['upstream_source_sha256']:raise ValueError('Frozen JevAny package changed')
    trainer_file=Path(cfg['trainer_config']);args=tomllib.loads(trainer_file.read_text())
    warm=cfg.get('initialization','fresh')=='continued'
    if (cfg.get('initialization','fresh') not in ('fresh','continued')
            or digest(trainer_file)!=cfg['trainer_config_sha256'] or args.get('resume')
            or (bool(args.get('init_from'))!=warm)):
        raise ValueError('Explicit fresh/continued trainer recipe required; no optimizer resume')
    if args.get('base')!=cfg['base'] or args.get('base_revision')!=cfg['base_revision']:
        raise ValueError('Pinned pretrained base identity differs')
    receipt_file=Path(cfg['data_receipt']);receipt=read(receipt_file)
    if digest(receipt_file)!=cfg['data_receipt_sha256'] or receipt['status']!='balanced_raw_data_ready':
        raise ValueError('Registered balanced raw data required')
    if not receipt['requires_masked_soft_target_loss'] or args.get('decision_mode','pointer')!='pointer' or args.get('rlcr',False):
        raise ValueError('Pointer SFT with registered masked soft targets required')
    inputs={str(receipt_file):cfg['data_receipt_sha256'],str(trainer_file):cfg['trainer_config_sha256'],**cfg['base_files']}
    for split,key in [('train','data'),('validation','eval_data')]:
        f=receipt['files'][split]
        if Path(args[key]).resolve()!=Path(f['path']).resolve():raise ValueError('Trainer data route differs')
        inputs[f['path']]=f['sha256']
    parent=None
    if warm:
        from .initialization import bind_parent
        parent=bind_parent(cfg,args)
        for name,checksum in parent['files'].items():
            if name in inputs and inputs[name]!=checksum:raise ValueError('Parent input identity conflicts')
            inputs[name]=checksum
    def verify():
        for name,checksum in inputs.items():
            if digest(Path(name))!=checksum:raise ValueError('Registered training input changed: '+name)
        if package_digest(upstream/'jevany')!=cfg['upstream_source_sha256']:raise ValueError('Upstream source changed')
    verify();out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    if Path(args['out']).resolve()!=out.resolve()/'training-run':raise ValueError('Unexpected trainer output route')
    if warm:write(out/'parent-initialization-private.json',parent)
    write(out/'protocol.json',dict(config=cfg,trainer_args=args,input_sha256=inputs,
        instruction=INSTRUCTION,initialization_mode='continued' if warm else 'fresh',parent_initialization=parent,
        initialization=('verified existing unified LoRA and pointer; new optimizer; pinned frozen base' if warm else
            'pinned pretrained base with freshly initialized LoRA and pointer'),
        selection='minimum phase-weighted hard/masked-soft target CE; fresh match verification required',
        code_sha256={str(p):digest(p) for p in Path(__file__).resolve().parents[1].rglob('*.py')},
        test_payloads_opened=False,deployed=False))
    sys.path.insert(0,str(upstream));from jevany import train as trainer
    if not Path(trainer.__file__).resolve().is_relative_to(upstream):raise ValueError('Unpinned upstream trainer imported')
    records=[json.loads(line) for line in Path(args['eval_data']).read_text().splitlines() if line.strip()]
    originals={key:getattr(trainer,key) for key in ('DecisionModel','question_loss','evaluate_during_training','training_requests','batch_loss')}
    original_warm_start=trainer.Checkpoint.warm_start
    curve=[];state=dict(micro_batches=0,gradient_proof_written=False);training_clock=[None]
    def construct(*pos,**kw):
        model=originals['DecisionModel'](*pos,**kw)
        adapters={n:p for n,p in model.lm.named_parameters() if 'lora_' in n}
        b=[p for n,p in adapters.items() if 'lora_B.' in n];a=[p for n,p in adapters.items() if 'lora_A.' in n]
        if (not a or not b or any(torch.count_nonzero(p).item() for p in b)
                or not any(torch.count_nonzero(p).item() for p in a)
                or any(p.requires_grad for n,p in model.lm.named_parameters() if 'lora_' not in n)
                or not all(p.requires_grad for p in adapters.values())):
            raise ValueError('Fresh LoRA initialization or frozen-base scope differs')
        if not warm:
            state['head_initial']={n:p.detach().cpu().clone() for n,p in model.head.named_parameters()}
            write(out/'initialization-audit.json',dict(fresh_lora_B_zero=True,fresh_lora_A_nonzero=True,
                pretrained_base_requires_grad=False,old_adapters_loaded=False,external_residual_networks=0,
                trainable_parameters=sum(p.numel() for p in model.trainable_parameters()),adapter_tensors=len(adapters)))
        return model
    def warm_start(checkpoint,model,meta):
        from peft import get_peft_model_state_dict
        from .initialization import exact_loaded_tensors
        if Path(checkpoint.path).resolve()!=Path(parent['model']['directory']).resolve():
            raise ValueError('Unexpected warm-start checkpoint')
        loaded=original_warm_start(checkpoint,model,meta)
        adapters=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
        exact=exact_loaded_tensors(adapters,model.head.state_dict(),parent['model']['directory'])
        state['adapter_initial']={n:p.detach().cpu().clone() for n,p in adapters.items()}
        state['head_initial']={n:p.detach().cpu().clone() for n,p in model.head.named_parameters()}
        write(out/'initialization-audit.json',dict(initialization='continued',**exact,
            parent_identity_sha256=parent['model']['identity_sha256'],parent_selected_step=parent['selected_step'],
            pretrained_base_requires_grad=False,old_adapters_loaded=True,external_residual_networks=0,
            trainable_parameters=sum(p.numel() for p in model.trainable_parameters()),
            new_optimizer=True,parent_group_proof=parent['group_proof']))
        return loaded
    def admitted(*pos,**kw):
        result=originals['training_requests'](*pos,**kw);report=result[1]
        if report['rejected_records'] or report['admitted_records']!=receipt['files']['train']['records']:
            raise ValueError('Token admission silently dropped prepared data')
        write(out/'token-admission.json',report);return result
    def evaluate(*pos,**kw):
        summary=originals['evaluate_during_training'](*pos,**kw)
        step=kw.get('step',pos[5] if len(pos)>5 else None)
        directory=Path(kw.get('out_dir',pos[4] if len(pos)>4 else ''))
        rows=read(directory/'training_eval'/f'step-{step:06d}'/'development'/'rows.json')
        metrics=validation_profile(records,rows,cfg['validation_phase_weights'])
        metrics.update(step=step,upstream_hard_label_nll=summary['clean']['nll'])
        curve.append(metrics);write(out/'validation-curve.json',curve)
        # The upstream on-disk benchmark retains its hard-label metric. Only
        # the in-memory checkpoint-selection metric uses registered target CE.
        result=copy.deepcopy(summary);result['clean']['hard_label_nll']=result['clean']['nll']
        result['clean']['nll']=metrics['weighted_target_ce']
        write(out/'status.json',dict(status='training_validation',**metrics));print(json.dumps(metrics),flush=True)
        return result
    def batch(*pos,**kw):
        model,a=pos[:2];state['micro_batches']+=1;micro=state['micro_batches'];steps=(micro-1)//a.accum
        if training_clock[0] is None:training_clock[0]=time.monotonic()
        if steps>=8 and not state['gradient_proof_written']:
            if warm:
                from peft import get_peft_model_state_dict
                actual=get_peft_model_state_dict(model.lm,save_embedding_layers=False)
                changed=any(not torch.equal(p.detach().cpu(),state['adapter_initial'][n]) for n,p in actual.items())
            else:changed=any(torch.count_nonzero(p).item() for n,p in model.lm.named_parameters() if 'lora_B.' in n)
            head_changed=any(not torch.equal(p.detach().cpu(),state['head_initial'][n]) for n,p in model.head.named_parameters())
            if not changed or not head_changed:raise ValueError('Actual raw-input LoRA/head updates missing')
            write(out/'actual-update-proof.json',dict(completed_optimizer_steps_at_least=steps,
                **({'continued_LoRA_parameters_changed':True} if warm else {'fresh_LoRA_B_now_nonzero':True}),
                head_parameters_changed=True,cached_encoder_features_used=False))
            state['gradient_proof_written']=True;del state['head_initial']
            state.pop('adapter_initial',None)
        if micro%64==1:
            write(out/'status.json',dict(status='training_raw_requests',micro_batches_started=micro,
                completed_optimizer_steps_at_least=steps,training_seconds=time.monotonic()-training_clock[0],
                actual_update_proof=state['gradient_proof_written'],max_steps=args['max_steps']))
        return originals['batch_loss'](*pos,**kw)
    trainer.DecisionModel=construct;trainer.question_loss=question_loss
    trainer.training_requests=admitted;trainer.evaluate_during_training=evaluate;trainer.batch_loss=batch
    if warm:trainer.Checkpoint.warm_start=warm_start
    write(out/'status.json',dict(status='loading_pinned_pretrained_base',LoRA_training_started=False))
    try:
        directory=trainer.main(['--config',str(trainer_file),'--base_load_path',cfg['base_load_path']])
        selection=read(directory/'selection.json');metrics=read(directory/'training_metrics.json');verify()
        selected=selection['selected_checkpoint'];checkpoint=(directory/selected).resolve() if selected else None
        result=dict(status='continued_base_LoRA_training_complete' if warm else 'fresh_base_LoRA_training_complete',model_id=cfg['model_id'],
            optimizer_steps=metrics['optimizer_steps'],selected_step=selection['step'],
            selected_checkpoint=str(checkpoint) if checkpoint else None,
            validation_curve=curve,actual_update_proof=state['gradient_proof_written'],
            pretrained_base_frozen=True,old_adapters_loaded=warm,external_residual_networks=0,
            fresh_full_match_improvement_proven=False,deployed=False)
        write(out/'result.json',result);write(out/'status.json',result)
    finally:
        for name,original in originals.items():setattr(trainer,name,original)
        trainer.Checkpoint.warm_start=original_warm_start


def main():
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    train(json.loads(p.parse_args().config.read_text()))
