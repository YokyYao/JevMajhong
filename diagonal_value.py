"""Low-capacity paired-point regression folded into the existing Jev pointer head."""
import copy
import math
import random

import torch


def pointer_products(head,decide,options):
    if head.residual is not None or head.temperature!=1.:
        raise ValueError('Diagonal calibration requires the plain temperature1 pointer head')
    return head.k(options)*head.q(decide).unsqueeze(0)*head.scale


def ridge_fit(x,y,alpha):
    if alpha<=0 or not math.isfinite(alpha) or len(x)!=len(y) or not len(x):
        raise ValueError('Ridge needs finite positive regularization and aligned data')
    x,y=x.double(),y.double()
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite paired data')
    # No intercept: identical action features must imply zero point difference.
    return torch.linalg.solve(x.T@x/len(x)+alpha*torch.eye(x.shape[1],dtype=x.dtype,device=x.device),x.T@y/len(x))


def wall_folds(seeds,folds=5,split_seed=2026103003):
    unique=sorted(set(seeds))
    if len(unique)<folds or folds<2:raise ValueError('Need multiple complete wall-stream folds')
    random.Random(split_seed).shuffle(unique)
    assignment={s:i%folds for i,s in enumerate(unique)}
    return [[i for i,s in enumerate(seeds) if assignment[s]==fold] for fold in range(folds)]


def cross_validate(x,y,seeds,alphas,folds=5):
    heldout=wall_folds(seeds,folds)
    reports=[]
    for alpha in alphas:
        error=0.;details=[]
        for indices in heldout:
            mask=torch.ones(len(x),dtype=torch.bool);mask[indices]=False
            coef=ridge_fit(x[mask],y[mask],alpha)
            residual=x[indices].double()@coef-y[indices].double()
            squared=float(residual.square().sum());error+=squared
            details.append({'train_streams':len({s for i,s in enumerate(seeds) if bool(mask[i])}),
                            'heldout_streams':len({seeds[i] for i in indices}),'states':len(indices),
                            'squared_error':squared})
        reports.append({'alpha':alpha,'mse':error/len(x),'folds':details})
    best=min(reports,key=lambda r:(r['mse'],-r['alpha']))
    zero=float(y.double().square().mean())
    return {'best_alpha':best['alpha'],'best_mse':best['mse'],'zero_predictor_mse':zero,
            'improves_over_zero':best['mse']<zero,'reports':reports,'split_seed':2026103003,
            'grouping':'entire logical wall stream; all seats/replicas together'}


def folded_head(head,coefficient,strength):
    if strength<0 or not math.isfinite(strength):raise ValueError('Invalid calibration strength')
    if head.residual is not None or head.temperature!=1.:raise ValueError('Unsupported source head')
    result=copy.deepcopy(head)
    factor=1+strength*coefficient.to(device=result.q.weight.device,dtype=result.q.weight.dtype)
    if factor.shape!=result.q.bias.shape or not torch.isfinite(factor).all():raise ValueError('Invalid diagonal coefficients')
    with torch.no_grad():
        result.q.weight.mul_(factor.unsqueeze(1));result.q.bias.mul_(factor)
    return result.eval()
