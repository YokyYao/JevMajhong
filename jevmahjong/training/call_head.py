"""Complete-option call-head repair, with train-only reference-policy anchors."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import torch

from jevmahjong.core.io import digest, write
from jevmahjong.data.native_pairs import load_sources, select_decisions
from jevmahjong.policy.routing import make_head
from .fit import save_head_checkpoint
from .provenance import snapshot, verify
from .sampling import state_key


def complete_groups(rows):
    """One decision is one unit, including every eligible alternative."""
    buckets = defaultdict(list)
    for row in rows: buckets[state_key(row)].append(row)
    groups = []
    for key, items in sorted(buckets.items()):
        first = items[0]; eligible = first['eligible']; teacher = first['teacher_action']
        if teacher not in eligible or len(eligible) != len(set(eligible)):
            raise ValueError('Invalid eligible menu')
        credit = {teacher: dict(utility=0., net_points=0., fast_reward=0., dealin=0.)}
        for row in items:
            if (row['eligible'] != eligible or row['teacher_action'] != teacher
                    or row['public_request'] != first['public_request']
                    or not torch.equal(row['reference'], first['reference'])
                    or not torch.equal(row['decide'], first['decide'])
                    or not torch.equal(row['options'], first['options'])
                    or row['action'] in credit):
                raise ValueError('Conflicting complete-option decision')
            c = row['training_credit']
            if c['control'] != first['training_credit']['control']:
                raise ValueError('Alternatives must share paired control futures')
            delta = {k: c['branch'][k] - c['control'][k] for k in credit[teacher]}
            if abs(delta['utility'] - row['training_advantage']) > 1e-12:
                raise ValueError('Credit and utility differ')
            credit[row['action']] = delta
        if set(credit) != set(eligible): raise ValueError('Every eligible option requires a label')
        q = first['reference'][eligible].double(); q = q / q.sum()
        if (not torch.isfinite(q).all() or (q < 0).any()
                or eligible[int(q.argmax())] != teacher): raise ValueError('Invalid prior')
        utility = torch.tensor([credit[a]['utility'] for a in eligible], dtype=torch.float64)
        if not torch.isfinite(utility).all(): raise ValueError('Nonfinite utility')
        groups.append(dict(key=key, row=first, eligible=eligible, teacher=teacher,
            prior=q, utility=utility, credit=credit))
    return groups


def logits(head, row):
    return head(row['decide'], row['options'])[row['eligible']].double()


def anchor_kl(head, row):
    q = row['reference'][row['eligible']].double(); q = q / q.sum()
    return (q * (q.clamp_min(1e-30).log() - logits(head, row).log_softmax(-1))).sum()


def distillation_loss(head, group, strength):
    if not math.isfinite(strength) or strength <= 0: raise ValueError('Positive prior strength required')
    target = (group['prior'].clamp_min(1e-30).log() + group['utility'] / strength).softmax(-1)
    return -(target * logits(head, group['row']).log_softmax(-1)).sum()


@torch.no_grad()
def profile(head, groups):
    if not groups: raise ValueError('Nonempty decision group required')
    records = []
    for g in groups:
        selected = g['eligible'][int(logits(head, g['row']).argmax())]
        c = g['credit'][selected]
        records.append(dict(label_regret=float(g['utility'].max()) - c['utility'],
            mean_label_utility_gain=c['utility'], mean_sampled_net_point_gain=c['net_points'],
            mean_sampled_fast_gain=c['fast_reward'], mean_sampled_dealin_change=c['dealin'],
            changed_argmax_states=int(selected != g['teacher'])))
    return dict(states=len(records), **{k: (sum(r[k] for r in records) if k == 'changed_argmax_states'
        else sum(r[k] for r in records) / len(records)) for k in records[0]})


def policy_improved(candidate, best, baseline, anchor_limit):
    """Loss-only improvement never resets the actual-choice early-stop clock."""
    p = candidate['held_policy']; b = best['held_policy']; start = baseline['held_policy']
    return (candidate['anchor_kl'] <= anchor_limit
        and p['mean_sampled_net_point_gain'] > b['mean_sampled_net_point_gain'] + 1e-12
        and p['mean_label_utility_gain'] > start['mean_label_utility_gain'] + 1e-12)


def route_groups(groups):
    from jevmahjong.policy.call_features import option_features
    result=defaultdict(list)
    for g in groups:
        req=g['row']['public_request'];state=req['state'];seat=state['seat']
        closed=all(m['type']==3 for m in state['melds'][seat]);features=option_features(req)
        criteria=req['questions']['action']['criteria'];names=['closed' if closed else 'open']
        if set(state['riichi'] or [])-{seat}:names.append('opponent_riichi')
        if closed and any(c['type']=='chi' and features[k]['shanten_gain']>0
                and features[k]['all_simples_material_after'] for k,c in criteria.items()):names.append('closed_chi_tanyao_gain')
        if closed and any(c['type']=='pon' and (features[k]['secured_dragon_han'] or features[k]['secured_wind_han'])
                for k,c in criteria.items()):names.append('closed_pon_yakuhai')
        for name in names:result[name].append(g)
    return dict(result)


@torch.no_grad()
def route_profile(head, groups):
    result=profile(head,groups);positive=recovers=negative=skipped=0
    for g in groups:
        values=list(g['row']['public_request']['questions']['action']['criteria'].values())
        selected=g['eligible'][int(logits(head,g['row']).argmax())]
        calls=[a for a in g['eligible'] if values[a]['type'] in ('chi','pon')]
        if values[g['teacher']]['type']=='none':
            good=[a for a in calls if g['credit'][a]['utility']>0 and g['credit'][a]['net_points']>0]
            positive+=bool(good);recovers+=selected in good
            bad=bool(calls) and all(g['credit'][a]['utility']<=0 and g['credit'][a]['net_points']<=0 for a in calls)
            negative+=bad;skipped+=bad and values[selected]['type']=='none'
    result.update(missed_positive_call_states=positive,recovered_sample_positive_call_states=recovers,
        all_sample_negative_call_states=negative,negative_states_still_skipped=skipped)
    return result


def load_inputs(cfg, protocol):
    audits = set(); audit_hashes = {}
    for name in cfg['required_data_audits']:
        root = Path(name).resolve(); registered = json.loads((root/'protocol.json').read_text())
        receipt = json.loads((root/'result.json').read_text()); verify(registered)
        if receipt['status'] != 'passed_complete' or registered['model_files'] != protocol['model_files']:
            raise ValueError('Complete matching input audit required')
        for path, sha in registered['source_files'].items():
            if digest(Path(path)) != sha: raise ValueError('Audited source changed')
        for file in ('protocol.json','result.json'): audit_hashes[str(root/file)] = digest(root/file)
        audits.add(Path(registered['config']['collection']).resolve())
    for path in cfg['data']:
        if Path(path).resolve().parent not in audits: raise ValueError('Every fit source needs its complete-option audit')
    parts, hashes, sources = load_sources(cfg['data'], protocol['model_files'], cfg['reward'])
    parts = select_decisions(parts, 'all', ['calling'])
    groups = {split: complete_groups(rows) for split, rows in parts.items()}
    if not all(groups.values()): raise ValueError('Whole-wall train/held groups required')
    if {g['key'][0] for g in groups['train']} & {g['key'][0] for g in groups['held']}:
        raise ValueError('Whole wall crosses training and validation')
    old, anchor_hashes, anchor_sources = load_sources(cfg['reference_anchor_data'], protocol['model_files'])
    anchors = {}
    for row in old['train']:
        if row['scope'] != 'calling': continue
        key = state_key(row)
        # Only forward inputs and original probabilities enter anchor training.
        anchors.setdefault(key, {k: row[k] for k in ('decide','options','eligible','reference')})
    old_held = {state_key(r) for r in old['held']}
    if set(anchors) & old_held: raise ValueError('Anchor state crosses old source split')
    def order(key): return hashlib.sha256(json.dumps([cfg['seed'],key]).encode()).hexdigest()
    keys = sorted(anchors, key=order)[:cfg['reference_anchor_states']]
    if len(keys) != cfg['reference_anchor_states']: raise ValueError('Insufficient distinct train-only anchors')
    train_walls = {g['key'][0] for split in groups.values() for g in split}
    if train_walls & {key[0] for key in keys}: raise ValueError('Anchor and new fit walls overlap')
    meta = dict(sources=anchor_sources, source_sha256=anchor_hashes, keys=[list(k) for k in keys],
                held_states_used=0, labels_used=False)
    return parts, groups, hashes, sources, [anchors[k] for k in keys], meta, audit_hashes


def fit(cfg):
    torch.set_num_threads(4); torch.manual_seed(cfg['seed'])
    if (cfg['train_scopes'] != ['calling'] or cfg['decision_filter'] != 'all'
            or cfg['fit_model'] != 'complete_option_call_head'
            or cfg['anchor_beta'] <= 0 or cfg['anchor_kl_limit'] <= 0
            or cfg['patience_epochs'] < 1): raise ValueError('Explicit anchored call-head fit required')
    if (any(not math.isfinite(cfg[k]) or cfg[k]<=0 for k in
            ('lr','anchor_beta','anchor_kl_limit','reference_prior_strength'))
            or any(not isinstance(cfg[k],int) or cfg[k]<1 for k in
                ('epochs','batch','anchor_batch','reference_anchor_states','patience_epochs'))
            or not math.isfinite(cfg['weight_decay']) or cfg['weight_decay']<0):
        raise ValueError('Finite positive fitting parameters required')
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=False)
    protocol = snapshot(cfg); write(out/'protocol.json',protocol)
    parts, groups, hashes, sources, anchors, anchor_meta, audit_hashes = load_inputs(cfg, protocol)
    routed={split:route_groups(gs) for split,gs in groups.items()}
    write(out/'anchor-inputs-private.json', anchor_meta)
    heads = {s: make_head(torch.load(Path(cfg[role+'_checkpoint'])/'head.pt', map_location='cpu', weights_only=True),'cpu')
        for s, role in [('ordinary','discard'),('calling','call')]}
    heads['ordinary'].requires_grad_(False); head = heads['calling']
    with torch.no_grad():
        for row in [g['row'] for gs in groups.values() for g in gs] + anchors:
            actual = head(row['decide'],row['options']).softmax(-1)
            if not torch.allclose(actual,row['reference'],atol=2e-5,rtol=2e-4): raise ValueError('Cached call encoder/head differs')
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    curve = []; updates = stale = 0; reason = 'epoch_limit'
    def save(epoch):
        held = profile(head, groups['held']); train = profile(head, groups['train'])
        with torch.no_grad():
            drift = sum(float(anchor_kl(head,r)) for r in anchors) / len(anchors)
            loss = sum(float(distillation_loss(head,g,cfg['reference_prior_strength'])) for g in groups['held'])/len(groups['held'])
        item = save_head_checkpoint(out, cfg['model_id'], heads, protocol, epoch, loss)
        item.update(held_policy=held, train_policy=train, anchor_kl=drift)
        item['held_routes']={name:route_profile(head,gs) for name,gs in routed['held'].items()}
        item['train_routes']={name:route_profile(head,gs) for name,gs in routed['train'].items()}
        curve.append(item); return item
    best = baseline = save(0)
    for epoch in range(1,cfg['epochs']+1):
        order = list(range(len(groups['train']))); rng = random.Random(cfg['seed']+epoch); rng.shuffle(order)
        anchor_order = list(range(len(anchors))); rng.shuffle(anchor_order)
        for start in range(0,len(order),cfg['batch']):
            batch = [groups['train'][i] for i in order[start:start+cfg['batch']]]
            begin = updates*cfg['anchor_batch']
            ar = [anchors[anchor_order[(begin+i)%len(anchors)]] for i in range(cfg['anchor_batch'])]
            loss = torch.stack([distillation_loss(head,g,cfg['reference_prior_strength']) for g in batch]).mean()
            loss = loss + cfg['anchor_beta']*torch.stack([anchor_kl(head,r) for r in ar]).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            optimizer.zero_grad(); loss.backward(); norm = torch.nn.utils.clip_grad_norm_(head.parameters(),.5)
            if not torch.isfinite(norm): raise ValueError('Nonfinite gradient')
            optimizer.step(); updates += 1
        item = save(epoch)
        if policy_improved(item,best,baseline,cfg['anchor_kl_limit']): best=item; stale=0
        else: stale += 1
        write(out/'status.json',dict(status='training',optimizer_updates=updates,curve=curve,selected=best))
        if stale >= cfg['patience_epochs']: reason='held_actual_choices_stalled'; break
    verify(protocol)
    for path, sha in {**hashes, **anchor_meta['source_sha256'], **audit_hashes}.items():
        if digest(Path(path)) != sha: raise ValueError('Fit or anchor input changed')
    result = dict(status='trained',model_id=cfg['model_id'],fit_method='complete-option-call-pointer-head-with-train-only-reference-KL',
        config=cfg,selected=best,curve=curve,optimizer_updates=updates,stop_reason=reason,trained_scopes=['calling'],decision_filter='all',
        train_states=len(groups['train']),held_states=len(groups['held']),train_rows=len(parts['train']),held_rows=len(parts['held']),
        data=cfg['data'],data_sources=sources,data_sha256=hashes,anchor_metadata=anchor_meta,data_audit_sha256=audit_hashes,
        selected_is_starting_model=best['epoch']==0,held_policy_gain_improved=best['epoch']!=0,
        human_test_opened=False,deployed=False,strength_improvement_proven=False)
    write(out/'result.json',result); write(out/'status.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    fit(json.loads(parser.parse_args().config.read_text()))


if __name__=='__main__': main()
