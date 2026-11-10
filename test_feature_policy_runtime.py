import copy
import unittest

from call_option_features import enrich_request
from clean_policy_data import policy_request
from clean_policy_runtime import CleanPolicyRouter
from encoder_call_policy import EncoderCallRouter
from feature_policy_runtime import FeatureClient
from jevany.data import api_request
from test_call_option_features import request


class Client:
    def __init__(self):self.requests=[]
    def system_one(self,**value):self.requests.append(copy.deepcopy(value));return {'ok':True}


class FeatureRuntimeTests(unittest.TestCase):
    def test_served_call_matches_enriched_training_input_exactly(self):
        ordinary,candidate=Client(),Client()
        router=EncoderCallRouter(ordinary,CleanPolicyRouter(ordinary,FeatureClient(candidate)))
        raw=request();router.system_one(**api_request(raw))
        self.assertEqual(candidate.requests,[api_request(enrich_request(policy_request(raw)))])
        self.assertEqual(len(ordinary.requests),0)
        self.assertNotIn('_meta',candidate.requests[0])
        self.assertNotIn('label',candidate.requests[0]['questions']['action'])

    def test_ordinary_policy_receives_original_prompt_and_options(self):
        ordinary,candidate=Client(),Client()
        router=EncoderCallRouter(ordinary,CleanPolicyRouter(ordinary,FeatureClient(candidate)))
        raw=request();raw['questions']['action']['criteria']={'a0':{'type':'dahai','pai':'1m'},'a1':{'type':'reach'}}
        sent=api_request(raw);router.system_one(**sent)
        self.assertEqual(ordinary.requests,[sent]);self.assertEqual(candidate.requests,[])


if __name__=='__main__':unittest.main()
