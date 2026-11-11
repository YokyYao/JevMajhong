"""Small nonlinear public-feature value model with a zero-output starting policy."""
import copy
import random
from pathlib import Path
import torch
from jevmahjong.core.io import digest,write
from jevmahjong.policy.kan_context import VERSION,NAMES
from .context_fit import prepare,encode,prediction_profile
from .sampling import weights
from .provenance import verify


def make_network(hidden):
    if not isinstance(hidden,int) or not 1 <= hidden <= 128:raise ValueError('Invalid hidden width')
    net=torch.nn.Sequential(torch.nn.Linear(len(NAMES),hidden),torch.nn.Tanh(),torch.nn.Linear(hidden,1,bias=False)).double()
    torch.nn.init.zeros_(net[2].weight)
    return net


def fit(cfg):
    out,protocol,parts,hashes,sources,heads=prepare(cfg);torch.manual_seed(cfg['seed'])
    matrices,_,y=encode(parts['train']);held_matrices,_,hy=encode(parts['held'])
    def sides(rows,matrices):
        return (torch.stack([m[r['action']] for m,r in zip(matrices,rows)]),
            torch.stack([m[r['teacher_action']] for m,r in zip(matrices,rows)]))
    xa,xt=sides(parts['train'],matrices);ha,ht=sides(parts['held'],held_matrices)
    w,coverage=weights(parts['train'],cfg['action_mix'],cfg['max_example_weight'],cfg.get('context_mix'),cfg.get('state_balanced_training',False))
    w=torch.tensor(w,dtype=torch.float64);write(out/'data-coverage.json',coverage)
    net=make_network(cfg['hidden']);optimizer=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    curve=[]
    @torch.no_grad()
    def save(epoch):
        prediction=(net(ha)-net(ht)).flatten();prof=prediction_profile(prediction,parts['held'],hy,sources,cfg.get('state_balanced_validation',False))
        checkpoint=out/f'{cfg["model_id"]}-e{epoch:02d}.pt'
        context={'version':VERSION,'names':list(NAMES),'model_kind':'mlp','hidden':cfg['hidden'],
            'state_dict':{k:v.detach().clone() for k,v in net.state_dict().items()}}
        model={'model_id':cfg['model_id'],'model_files':protocol['model_files'],'base_and_LoRA_frozen':True,
            'heads':{s:copy.deepcopy(h.state_dict()) for s,h in heads.items()},'trained_scopes':['ordinary'],
            'decision_filter':'kan_windows','kan_context':context,'epoch':epoch}
        torch.save(model,checkpoint)
        item={'epoch':epoch,'held_loss':prof[-1]['loss'],'validation_sources':prof,'checkpoint':str(checkpoint.resolve()),
            'sha256':digest(checkpoint),'train_mse':float(((net(xa)-net(xt)).flatten()-y).square().mean())}
        curve.append(item);return item
    best=save(0);baseline=best['held_loss'];stale=updates=0;reason='epoch_limit'
    for epoch in range(1,cfg['epochs']+1):
        order=list(range(len(y)));random.Random(cfg['seed']+epoch).shuffle(order)
        for start in range(0,len(order),cfg['batch']):
            indexes=order[start:start+cfg['batch']];prediction=(net(xa[indexes])-net(xt[indexes])).flatten()
            loss=(w[indexes]*(prediction-y[indexes]).square()).mean()/w.mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite context loss')
            optimizer.zero_grad();loss.backward();gradient=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            if not torch.isfinite(gradient):raise ValueError('Nonfinite context gradient')
            optimizer.step();updates+=1
        item=save(epoch)
        if item['held_loss']<best['held_loss']:best=item;stale=0
        else:stale+=1
        write(out/'status.json',{'status':'training','updates':updates,'curve':curve,'selected':best})
        if stale>=cfg['patience_epochs']:reason='held_loss_stalled';break
    verify(protocol)
    for path,h in hashes.items():
        if digest(Path(path))!=h:raise ValueError('Input changed during fit')
    result={'status':'trained','model_id':cfg['model_id'],'selected':best,'curve':curve,'config':cfg,
        'fit_method':'nonlinear-public-feature-paired-utility-MSE',
        'loss_definition':'state-balanced mean squared paired utility error' if cfg.get('state_balanced_validation') else 'mean squared paired utility error',
        'parameter_count':sum(p.numel() for p in net.parameters()),'optimizer_updates':updates,'stop_reason':reason,
        'train_rows':len(parts['train']),'held_rows':len(parts['held']),'data':[s['path'] for s in sources],
        'data_sources':sources,'data_sha256':hashes,'baseline_held_loss':baseline,
        'held_loss_improved':best['held_loss']<baseline,'selected_is_starting_model':best['epoch']==0,
        'selection_validation':'latest_source','human_test_opened':False,'deployed':False,'strength_improvement_proven':False}
    write(out/'result.json',result);write(out/'status.json',result)
