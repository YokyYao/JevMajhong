"""Recommendation loader matching the evaluated unmerged, temperature1 policy."""
from arena import ModelAgent


class NativeModelAgent(ModelAgent):
    def __init__(self,checkpoint,guarded=False):
        from jevany import JevModel
        from jevany.checkpoint import LoadOptions
        from jevany.inference import InferenceOptions
        self.model=JevModel.from_pretrained(checkpoint,device='cuda',dtype='bf16',
            options=LoadOptions(merge=False,temperature=1.),
            inference_options=InferenceOptions(max_state_tokens=4096,max_branch_tokens=8192,max_packed_tokens=8192))
        self.guarded=guarded
        self.calls=0
