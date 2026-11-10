"""Call-specific encoder adaptation; ordinary decisions use the incumbent."""
import copy
from call_policy import METADATA,is_call_request


class EncoderCallRouter:
    def __init__(self,ordinary,calling):
        self.ordinary,self.calling=ordinary,calling

    def system_one(self,**request):
        if not is_call_request(request):return self.ordinary.system_one(**request)
        request=copy.deepcopy(request)
        screen=request['state'].get('screen_observation',{})
        if screen and not screen.get('metadata_confirmed',False):
            confirmed=set(screen.get('confirmed_metadata',[]))
            for field in METADATA:
                if field not in confirmed:request['state'][field]=None
        return self.calling.system_one(**request)
