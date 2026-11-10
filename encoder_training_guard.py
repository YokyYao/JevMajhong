"""Exact chi/pon recall protection for encoder-adaptation validation."""
from collections import Counter,defaultdict
import math


def exact_metrics(records,rows):
    lookup={r['_meta']['id']:r for r in records}
    if len(lookup)!=len(records) or len(rows)!=len(lookup):raise ValueError('Validation coverage differs')
    seen=set();groups=defaultdict(lambda:{'n':0,'nll_sum':0.,'support':Counter(),'exact_tp':Counter()})
    for row in rows:
        identity=row['id']
        if identity in seen or identity not in lookup:raise ValueError('Duplicate/foreign validation prediction')
        seen.add(identity);record=lookup[identity];q=record['questions']['action'];keys=list(q['criteria'])
        if row['keys']!=keys or row['label']!=keys.index(q['label']):raise ValueError('Validation option alignment differs')
        p=row['p']
        if len(p)!=len(keys) or any(not math.isfinite(float(x)) or x<0 for x in p):raise ValueError('Invalid probabilities')
        if not math.isclose(sum(p),1.,rel_tol=1e-5,abs_tol=1e-5):raise ValueError('Unnormalized validation probabilities')
        pred=max(range(len(p)),key=lambda i:p[i]);gold=keys.index(q['label']);kind=q['criteria'][q['label']]['type']
        values=groups[record['_meta']['view']];values['n']+=1;values['support'][kind]+=1
        if pred==gold:values['exact_tp'][kind]+=1
        values['nll_sum']-=math.log(max(p[gold],1e-30))
    return {view:{'n':g['n'],'nll':g['nll_sum']/g['n'],'support':dict(g['support']),
                  'exact_tp':dict(g['exact_tp'])} for view,g in groups.items()}


class ExactRecallGuard:
    def __init__(self,views,fraction=.5):
        self.views=tuple(views);self.fraction=fraction;self.baseline=None;self.admissible=True;self.last={}

    def assess(self,metrics):
        if set(metrics)!=set(self.views):raise ValueError('Missing validation view')
        if self.baseline is None:
            self.baseline=metrics;self.admissible=True;self.last={'admissible':True,'warm_start':True,'reasons':[]};return self.last
        reasons=[]
        for view in self.views:
            for kind in ('chi','pon'):
                before=self.baseline[view]['exact_tp'].get(kind,0);now=metrics[view]['exact_tp'].get(kind,0)
                if now<max(1,math.ceil(before*self.fraction)):
                    reasons.append(f'{view}/{kind}: exact_tp {now} below protected warm-start fraction')
        self.admissible=not reasons;self.last={'admissible':self.admissible,'warm_start':False,'reasons':reasons}
        return self.last

    def update(self,original,value,best,stale,maximize=True,min_delta=0.):
        if self.admissible:return original(value,best,stale,maximize=maximize,min_delta=min_delta)
        # Report the actual NLL unchanged, but do not select a collapsed policy.
        return (value if best is None else best),stale+1,False
