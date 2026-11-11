"""Unconditional stage2 policy for a fair full-policy versus call-only ablation."""
from clean_policy_data import policy_request
from call_option_features import enrich_request


class FullStage2Client:
    """All model-scored actions use stage2; no ordinary or rare-action oldB fallback.

    Automatic wins and the common discard guard remain in the arena Agent for
    all three arms, exactly as in the previous strength assessments.
    """
    def __init__(self, candidate): self.candidate = candidate

    def system_one(self, **request):
        from jevany.data import api_request
        return self.candidate.system_one(**api_request(enrich_request(policy_request(request))))
