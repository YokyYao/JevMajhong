"""Fit a regularized public-feature value-difference component on native pairs."""
import argparse
import copy
import json
from pathlib import Path
import torch
from jevmahjong.core.io import digest, write
from jevmahjong.data.native_pairs import load_sources, select_decisions
from jevmahjong.policy.routing import make_head
from jevmahjong.policy.kan_context import VERSION, NAMES, feature_vectors
from .provenance import snapshot, verify
from .sampling import weights,state_key


def encode(rows):
    matrices = [torch.tensor(feature_vectors(r['public_request']), dtype=torch.float64) for r in rows]
    x = torch.stack([m[r['action']] - m[r['teacher_action']] for m, r in zip(matrices, rows)])
    y = torch.tensor([r['training_advantage'] for r in rows], dtype=torch.float64)
    return matrices, x, y


def profile(coefficients, rows, x, y, sources, state_balanced=False):
    return prediction_profile(x @ coefficients,rows,y,sources,state_balanced)


def prediction_profile(prediction,rows,y,sources,state_balanced=False):
    result = []
    for source in sources:
        mask = torch.tensor([source['seed_start'] <= r['episode_key'][0] < source['seed_start'] + source['seeds'] for r in rows])
        if not mask.any(): raise ValueError('Every source needs internal validation')
        yy, pp = y[mask], prediction[mask]
        errors=(pp-yy).square();absolute=(pp-yy).abs();extra={}
        if state_balanced:
            groups={}
            for i,r in enumerate([r for r,m in zip(rows,mask.tolist()) if m]):groups.setdefault(state_key(r),[]).append(i)
            errors=torch.stack([errors[indexes].mean() for indexes in groups.values()])
            absolute=torch.stack([absolute[indexes].mean() for indexes in groups.values()])
            extra={'states':len(groups),'aggregation':'equal public-state means within source'}
        result.append({'source':source['path'], 'rows':int(mask.sum()),
            'loss':float(errors.mean()), 'mae':float(absolute.mean()),**extra,
            'positive_pair_order_correct':int(((yy > 0) & (pp > 0)).sum()),
            'positive_pairs':int((yy > 0).sum()),
            'negative_pair_order_correct':int(((yy < 0) & (pp < 0)).sum()),
            'negative_pairs':int((yy < 0).sum())})
    return result


def prepare(cfg):
    out=Path(cfg['out']);out.mkdir(parents=True,exist_ok=False)
    if cfg['train_scopes'] != ['ordinary'] or cfg['decision_filter'] != 'kan_windows':raise ValueError('Kan-only context component required')
    if cfg.get('selection_validation','latest_source') != 'latest_source':raise ValueError('Select on latest source')
    torch.set_num_threads(4);protocol=snapshot(cfg);write(out/'protocol.json',protocol)
    parts,hashes,sources=load_sources(cfg['data'],protocol['model_files'],cfg['reward'])
    parts=select_decisions(parts,'kan_windows',['ordinary']);train,held=parts['train'],parts['held']
    if not train or not held:raise ValueError('Whole-wall train and validation required')
    heads={s:make_head(torch.load(Path(cfg[k])/'head.pt',map_location='cpu',weights_only=True),'cpu').eval()
        for s,k in [('ordinary','discard_checkpoint'),('calling','call_checkpoint')]}
    with torch.no_grad():
        for r in train+held:
            p=heads[r['scope']](r['decide'],r['options']).softmax(-1)
            if not torch.allclose(p,r['reference'],atol=2e-5,rtol=2e-4):raise ValueError('Original reference input or policy changed')
    return out,protocol,parts,hashes,sources,heads


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    cfg=json.loads(parser.parse_args().config.read_text())
    if cfg.get('fit_model') == 'residual_policy':
        from .context_policy import fit
        fit(cfg);return
    if cfg.get('feature_version', VERSION) != VERSION:
        raise ValueError('Progress feature schema requires residual_policy fitting')
    if cfg.get('fit_model','ridge') == 'mlp':
        from .context_network import fit
        fit(cfg);return
    if cfg.get('fit_model','ridge') != 'ridge':raise ValueError('Unknown context fit model')
    regularizers=cfg['regularizers']
    if not regularizers or len(set(regularizers)) != len(regularizers) or any(not 0 < v < float('inf') for v in regularizers):
        raise ValueError('Finite positive unique ridge strengths required')
    out,protocol,parts,hashes,sources,heads=prepare(cfg);train,held=parts['train'],parts['held']
    _,x,y=encode(train);_,hx,hy=encode(held)
    w,coverage=weights(train,cfg['action_mix'],cfg['max_example_weight'],cfg.get('context_mix'),cfg.get('state_balanced_training',False))
    w=torch.tensor(w,dtype=torch.float64);total=w.sum();write(out/'data-coverage.json',coverage)
    curve=[]
    def save(coef,regularization):
        prof=profile(coef,held,hx,hy,sources,cfg.get('state_balanced_validation',False));label='start' if regularization is None else f'ridge-{regularization:g}'
        path=out/f'{cfg["model_id"]}-{label}.pt'
        model={'model_id':cfg['model_id'],'model_files':protocol['model_files'],'base_and_LoRA_frozen':True,
            'heads':{s:copy.deepcopy(h.state_dict()) for s,h in heads.items()},
            'trained_scopes':['ordinary'],'decision_filter':'kan_windows',
            'kan_context':{'version':VERSION,'names':list(NAMES),'coefficients':coef.clone()}}
        torch.save(model,path)
        item={'regularization':regularization,'held_loss':prof[-1]['loss'],'validation_sources':prof,
            'checkpoint':str(path.resolve()),'sha256':digest(path),'train_mse':float((x@coef-y).square().mean())}
        curve.append(item);return item
    best=save(torch.zeros(len(NAMES),dtype=torch.float64),None);baseline=best['held_loss']
    normal=x.T@(w[:,None]*x)/total;rhs=x.T@(w*y)/total
    for strength in regularizers:
        coefficients=torch.linalg.solve(normal+float(strength)*torch.eye(len(NAMES),dtype=torch.float64),rhs)
        if not torch.isfinite(coefficients).all():raise ValueError('Nonfinite fitted coefficients')
        item=save(coefficients,strength)
        if item['held_loss'] < best['held_loss']:best=item
        write(out/'status.json',{'status':'fitting','solves':len(curve)-1,'selected':best})
    verify(protocol)
    for path,h in hashes.items():
        if digest(Path(path)) != h:raise ValueError('Input changed during fit')
    result={'status':'trained','model_id':cfg['model_id'],'selected':best,'curve':curve,'config':cfg,
        'fit_method':'weighted-ridge-paired-utility-differences','loss_definition':'mean squared paired utility error',
        'coefficient_count':len(NAMES),'optimizer_updates':0,'linear_solves':len(regularizers),
        'train_rows':len(train),'held_rows':len(held),'data':[s['path'] for s in sources],
        'data_sources':sources,'data_sha256':hashes,
        'baseline_held_loss':baseline,'held_loss_improved':best['held_loss']<baseline,
        'selected_is_starting_model':best['regularization'] is None,'selection_validation':'latest_source',
        'human_test_opened':False,'deployed':False,'strength_improvement_proven':False}
    write(out/'result.json',result);write(out/'status.json',result)


if __name__=='__main__':main()
