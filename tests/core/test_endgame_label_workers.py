import copy
import unittest
from tools.label_reused_endgame import partition_rows,validate_terminal_menu,frozen_protocol


class LabelWorkerTests(unittest.TestCase):
    def test_public_cost_partition_is_disjoint_and_future_independent(self):
        rows=[{'seed':i,'request':{'questions':{'action':{'criteria':{str(j):{} for j in range(n)}}}}}
              for i,n in enumerate((15,14,13,12,3,2,2,2))]
        bins=partition_rows(rows)
        self.assertEqual(sorted(r['seed'] for b in bins for r in b),list(range(8)))
        costs=[sum(len(r['request']['questions']['action']['criteria']) for r in b) for b in bins]
        self.assertLessEqual(abs(costs[0]-costs[1]),2)
        changed=copy.deepcopy(rows)
        for row in changed:row['future_rank']=1
        self.assertEqual([[r['seed'] for r in b] for b in bins],[[r['seed'] for r in b] for b in partition_rows(changed)])

    def test_verifier_rejects_incomplete_worlds_wrong_identity_or_settlement(self):
        row={'seat':3,'request_sha256':'source'}
        outcome={'ledger_passed':True,'scores':[40000,30000,20000,10000],'ranks':[1,2,3,4],'utility':-1.}
        label={'status':'complete_terminal_menu','source':row,'worlds_per_action':32,
            'source_request_sha256':'source','continued_until_native_match_done':True,
            'baseline_identity_sha256':'identity','actual_actions':2,
            'record':{'questions':{'action':{'criteria':{'win':{},'pass':{}}}}},
            'outcomes':[[copy.deepcopy(outcome) for _ in range(32)] for _ in range(2)]}
        self.assertEqual(validate_terminal_menu(label,row,'identity'),64)
        bad=copy.deepcopy(label);bad['outcomes'][1].pop()
        with self.assertRaises(ValueError):validate_terminal_menu(bad,row,'identity')
        bad=copy.deepcopy(label);bad['outcomes'][0][0]['ranks']=[4,3,2,1]
        with self.assertRaises(ValueError):validate_terminal_menu(bad,row,'identity')
        bad=copy.deepcopy(label);bad['outcomes'][0][0]['ledger_passed']=False
        with self.assertRaises(ValueError):validate_terminal_menu(bad,row,'identity')
        with self.assertRaises(ValueError):validate_terminal_menu(label,row,'another')

    def test_unregistered_continuation_is_rejected_before_source_reads(self):
        config={'out':'unused','schema':'reused-fourth-label32-v1','future_worlds':32,
                'continuation_policy':'unqualified_candidate','label_registered':True,'fit_registered':False,'workers':2}
        with self.assertRaises(ValueError):frozen_protocol(config)


if __name__=='__main__':unittest.main()
