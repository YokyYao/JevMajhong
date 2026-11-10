"""Select feature-policy candidates by actual NLL after minimal call-skill admission."""
class FeatureSkillGuard:
    def __init__(self,views):
        self.views=tuple(views);self.step=0;self.admissible=False

    def assess(self,metrics):
        if set(metrics)!=set(self.views):raise ValueError('Validation view coverage differs')
        reasons=[]
        for view in self.views:
            for kind in ('chi','pon'):
                if metrics[view]['exact_tp'].get(kind,0)<1:
                    reasons.append(f'{view}/{kind}: no exact hit')
        self.admissible=not reasons
        return {'admissible':self.admissible,'feature_baseline':self.step==0,'reasons':reasons}

    def update(self,original,value,best,stale,maximize=True,min_delta=0.):
        if self.admissible:return original(value,best,stale,maximize=maximize,min_delta=min_delta)
        # A low-loss all-pass baseline cannot block a later skilled candidate.
        return best,stale+(self.step>0),False
