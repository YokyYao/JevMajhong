"""Complete fixed-budget fits without pretending upstream used CE early stopping."""
import math
from pathlib import Path


def fixed_budget_selection(cfg,args,metrics,curve):
    """CE summary only; the predeclared final checkpoint remains the game candidate."""
    step=cfg.get('registered_final_step')
    if (cfg.get('evaluation_candidate_rule')!='final_registered_budget_v1'
            or isinstance(step,bool) or not isinstance(step,int) or step<=0
            or args.get('max_steps')!=step or args.get('early_stop_patience',0)!=0
            or metrics.get('optimizer_steps')!=step or metrics.get('early_stopped') is not False):
        raise ValueError('Completed explicitly registered fixed budget required')
    interval=args.get('eval_every_steps')
    if not isinstance(interval,int) or interval<=0 or not args.get('eval_before_start'):
        raise ValueError('Initial and periodic validation required')
    expected=sorted({0,step,*range(interval,step+1,interval)})
    if ([row.get('step') for row in curve]!=expected
            or any(not math.isfinite(row['weighted_target_ce']) for row in curve)):
        raise ValueError('Complete ordered fixed-budget validation trace required')
    best=min(curve,key=lambda row:row['weighted_target_ce'])
    return dict(selected_checkpoint=f"checkpoints/step-{best['step']:06d}",step=best['step'],
        value=best['weighted_target_ce'],metric='development_nll',
        record_origin='wrapper_fixed_budget_CE_summary',upstream_early_stopping_used=False,
        evaluation_candidate_rule=cfg['evaluation_candidate_rule'],registered_final_step=step)


def initial_snapshot(model,tokenizer,args,trainer,directory,step):
    """Serialize true pre-update parameters when upstream won't save step zero."""
    if step!=0 or args.get('early_stop_patience',0):return
    target=Path(directory)/'checkpoints/step-000000'
    target.mkdir(parents=True,exist_ok=False)
    model.lm.save_pretrained(target,save_embedding_layers=False)
    meta=trainer.Meta(base=args['base'],base_revision=args['base_revision'],lora=args['lora'],
        head=model.head.state_dict(),head_dim=args.get('head_dim',256),
        head_residual_dim=args.get('head_residual_dim',0),
        head_type='residual' if args.get('head_residual_dim',0) else 'linear',
        option_isolation=bool(args.get('option_isolation',False)),
        special_embeddings=model.special_embeddings,multimodal=args.get('multimodal',False),
        backbone_adapter=model.backbone_adapter,branch_mode=model.branch_mode,
        tokenizer_saved=True,weights_dtype=args.get('weights_dtype','fp32'),
        decision_mode=args.get('decision_mode','pointer'),verbalizers=model.verbalizers,
        extra={'args':args,'optimizer_step':0,'snapshot_origin':'wrapper_before_first_optimizer_update'})
    trainer.write_meta(target,meta);tokenizer.save_pretrained(target)


def completed_result(cfg,metrics,selection,curve,actual_update_proof):
    warm=cfg.get('initialization','fresh')=='continued'
    return dict(status='continued_base_LoRA_training_complete' if warm else 'fresh_base_LoRA_training_complete',
        model_id=cfg['model_id'],optimizer_steps=metrics['optimizer_steps'],selected_step=selection['step'],
        selected_checkpoint=str((Path(cfg['out'])/'training-run'/selection['selected_checkpoint']).resolve()),
        validation_curve=curve,actual_update_proof=actual_update_proof,
        pretrained_base_frozen=True,old_adapters_loaded=warm,external_residual_networks=0,
        fresh_full_match_improvement_proven=False,deployed=False)
