import copy
import json
from pathlib import Path
import unittest
from tools.diagnose_endgame_self_continuation import select_manifest,validate_configuration,validate_cached_outcome


def row(seed,rank,step=0):
    return {'seed':seed,'stratum':rank,'cohort':'standard','step':step,'seat':0,'request_sha256':str(seed)}


class ConsumedDiagnosticTests(unittest.TestCase):
    def test_public_sampler_preserves_distinct_sources_and_future_outcomes_do_not_select(self):
        rows=[row(1,'first'),row(1,'second'),row(2,'first')]
        selected=select_manifest(rows,{'first':1,'second':1},'fixed')
        self.assertEqual({(r['seed'],r['stratum']) for r in selected},{(2,'first'),(1,'second')})
        changed=[dict(r,future_score=999999 if r['seed']==1 else -999999) for r in rows]
        self.assertEqual([(r['seed'],r['stratum']) for r in selected],
                         [(r['seed'],r['stratum']) for r in select_manifest(changed,{'first':1,'second':1},'fixed')])

    def test_insufficient_source_coverage_is_not_replaced_by_repeated_windows(self):
        with self.assertRaises(ValueError):
            select_manifest([row(1,'first'),row(1,'first',1)],{'first':2},'fixed')

    def test_diagnostic_configuration_cannot_silently_enable_promotion(self):
        path=Path(__file__).resolve().parents[2]/'configs/training/endgame-consumed-continuation-diagnostic-v1.json'
        config=json.loads(path.read_text());validate_configuration(config)
        for name in ('fit_registered','promotion_registered','independent_confirmation','export_registered','raw_publication'):
            with self.assertRaises(ValueError):validate_configuration(dict(config,**{name:True}))

    def test_cached_baseline_must_agree_with_actual_score_placement_and_points(self):
        row={'seat':2,'baseline':{'terminal_ledger_passed':True,'final_scores':[40000,30000,20000,10000],
            'start_scores':[25000]*4,'rank':3,'utility':0.,'net_points_from_window':-5000}}
        validate_cached_outcome(row)
        for name,value in (('terminal_ledger_passed',False),('rank',2),('utility',1.),('net_points_from_window',5000)):
            bad=copy.deepcopy(row);bad['baseline'][name]=value
            with self.assertRaises(ValueError):validate_cached_outcome(bad)


if __name__=='__main__':unittest.main()
