"""Shared feature transformation for candidate serving; ordinary router is unchanged."""
from call_option_features import enrich_request


class FeatureClient:
    def __init__(self,client):self.client=client

    def system_one(self,**request):
        from jevany.data import api_request
        return self.client.system_one(**api_request(enrich_request(request)))
