import copy
import unittest

from clean_policy_data import POLICY_INSTRUCTION
from clean_policy_runtime import CleanPolicyRouter
from clean_policy_evaluation import aggregate
from guide_development_audit import interval


class Client:
    def __init__(self,name):self.name=name;self.requests=[]
    def system_one(self,**request):
        self.requests.append(copy.deepcopy(request))
        return {'client':self.name}


def request(types):
    return {'state':{'seat':0,'melds':[[],[],[],[]],'honba':3,'sticks':2,'scores':[25000]*4},
        'questions':{'action':{'instructions':'original','type':'choice',
            'criteria':{f'a{i}':{'type':k} for i,k in enumerate(types)},'label':'a0'}},
        '_meta':{'private':'must not enter inference'}}


def fixtures():
    cfg={'seed_start':100,'seeds':2,'stage':'development'}
    rows=[]
    for suite in ('closed','mixed','strong'):
        for seed,delta in ((100,100),(101,300)):
            for seat in range(4):
                for name in ('reference','candidate'):
                    gain=delta if name=='candidate' else 0
                    rows.append({'suite':suite,'seed':seed,'seat':seat,'policy':name,'score':25000+gain,
                        'rank':2,'hands':4,'hand_wins':1,'dealins':0,'calls':2,
                        'point_components':{'settlements':gain},'settlement_ledger_passed':True})
    return rows,cfg


class CleanPolicyEvaluationTests(unittest.TestCase):
    def test_discard_riichi_and_call_phases_use_clean_model_without_label_or_metadata(self):
        ordinary,candidate=Client('old'),Client('new');router=CleanPolicyRouter(ordinary,candidate)
        for types in [('dahai','dahai'),('dahai','reach'),('chi','pon','none')]:
            original=request(types);before=copy.deepcopy(original)
            self.assertEqual(router.system_one(**original)['client'],'new')
            sent=candidate.requests[-1]
            self.assertEqual(sent['questions']['action']['instructions'],POLICY_INSTRUCTION)
            self.assertNotIn('_meta',sent)
            self.assertNotIn('label',sent['questions']['action'])
            self.assertEqual(original,before)
        self.assertEqual(len(ordinary.requests),0)

    def test_rare_kan_options_use_unchanged_old_model_prompt(self):
        ordinary,candidate=Client('old'),Client('new');router=CleanPolicyRouter(ordinary,candidate)
        for types in [('dahai','ankan'),('pon','daiminkan','none'),('hora','none')]:
            self.assertEqual(router.system_one(**request(types))['client'],'old')
            self.assertEqual(ordinary.requests[-1]['questions']['action']['instructions'],'original')
        self.assertEqual(len(candidate.requests),0)

    def test_unknown_screenshot_uses_only_explicitly_confirmed_counters(self):
        ordinary,candidate=Client('old'),Client('new');router=CleanPolicyRouter(ordinary,candidate)
        r=request(('dahai','reach'))
        r['state']['screen_observation']={'metadata_confirmed':False,'confirmed_metadata':['honba','sticks']}
        router.system_one(**r)
        state=candidate.requests[0]['state']
        self.assertEqual((state['honba'],state['sticks']),(3,2))
        self.assertIsNone(state['scores'])
        self.assertEqual(r['state']['scores'],[25000]*4)

    def test_cluster_by_wall_instead_of_seats_or_suites(self):
        rows,cfg=fixtures();report=aggregate(rows,cfg);paired=report['overall']['candidate_minus_reference']
        self.assertEqual(report['games'],48)
        self.assertEqual(paired['independent_wall_streams'],2)
        self.assertEqual(paired['gain'],200)
        self.assertEqual(paired['ci95'],interval([100,300]))
        self.assertTrue(report['positive_signal'])
        self.assertFalse(report['ci_excludes_zero_positive'])

    def test_missing_or_duplicate_games_cannot_produce_complete_summary(self):
        rows,cfg=fixtures()
        with self.assertRaises(ValueError):aggregate(rows[:-1],cfg)
        with self.assertRaises(ValueError):aggregate(rows+[rows[0]],cfg)

    def test_wrong_terminal_point_ledger_and_nan_are_rejected(self):
        rows,cfg=fixtures();rows[0]['score']+=100
        with self.assertRaises(ValueError):aggregate(rows,cfg)
        rows,cfg=fixtures();rows[0]['score']=float('nan')
        with self.assertRaises(ValueError):aggregate(rows,cfg)


if __name__=='__main__':unittest.main()
