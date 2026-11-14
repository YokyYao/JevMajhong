"""One fresh-base adapter and pointer head for every decision window."""
from jevmahjong.data.base_lora_pool import normalized_request


def serving_request(request):
    record=normalized_request(request,'inference','inference',0)
    return dict(state=record['state'],questions={name:{k:v for k,v in q.items()
        if k in ('type','instructions','criteria')} for name,q in record['questions'].items()})


class UnifiedPolicyClient:
    """No incumbent encoder, head or external residual is composed here."""
    def __init__(self,model):self.model=model
    def system_one(self,**request):return self.model.system_one(**serving_request(request))
