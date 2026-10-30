import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from head_rl import fingerprint,write
from scripts.audit_diagonal_value import audit_development
from scripts.audit_score_confirmation import compare,summarize


class NativeDevelopmentAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name);self.source=self.root/'run';self.native=self.source/'native-development'
        self.native.mkdir(parents=True)
        self.config={'development_seed_start':10,'development_seeds':2,'reward_scale':1000.,
                     'target_mean_net_points':0,'target_net_points_gain':0}
        protocol={'config':self.config,'sha256':hashlib.sha256(json.dumps(self.config,sort_keys=True).encode()).hexdigest(),
                  'code_sha256':{}}
        write(self.source/'protocol.json',protocol)
        self.candidates=[];self.results=[]
        def games(points):
            return [{'seed':s,'seat':seat,'engine_seed':s*(1<<32),'seed_scheme':'spaced-v1',
                     'game_mode':'4p-red-east','reward_mode':'net_points','reward_scale':1000.,
                     'score':points,'rank':2,'reward':(points-25000)/1000.,'sampling_seed':None,
                     'trainable_decisions':0,'old_opponent':False} for s in (10,11) for seat in range(4)]
        def write_games(name,rows):
            directory=self.native/name;directory.mkdir()
            (directory/'games.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        baseline=games(25000);write_games('baseline',baseline)
        for index,points in enumerate((25100,25050),1):
            checkpoint=self.root/f'head-{index}';checkpoint.mkdir();(checkpoint/'head.pt').write_bytes(bytes([index]))
            item={'checkpoint':checkpoint.name,'head_sha256':fingerprint(checkpoint/'head.pt')}
            candidate=games(points);write_games(f'variant-{index:02d}',candidate)
            comparison=compare(candidate,baseline,{**self.config,'confirmation_seed_start':10,'confirmation_seeds':2})
            self.candidates.append(item)
            self.results.append({'variant':index,**item,'native_parity_checks':1,'metrics':summarize(candidate),
                                 'paired_exploratory':{key:comparison[key] for key in ('primary_delta','primary_delta_ci95')}})
        write(self.source/'training-result.json',{'candidates':self.candidates})
        write(self.native/'protocol.json',{'config':self.config,'source_protocol_sha256':protocol['sha256'],
                                         'code_sha256':{},'preselected_candidates':self.candidates})
        self.selection={'status':'complete','results':self.results,'baseline':summarize(baseline),
                        'selected':self.results[0],'independent_improvement_confirmed':False}
        self.persist()
        patcher=patch('scripts.audit_diagonal_value.ROOT',self.root);patcher.start();self.addCleanup(patcher.stop)

    def persist(self):
        write(self.native/'development-results.json',self.results);write(self.native/'selection.json',self.selection)

    def test_small_positive_gain_is_retained_without_fixed_gain_gate(self):
        result=audit_development(self.source)
        self.assertEqual(result['audit_status'],'passed')
        self.assertEqual(result['selected']['paired_exploratory']['primary_delta'],100.)

    def test_false_best_candidate_is_rejected(self):
        self.selection['selected']=self.results[1];self.persist()
        with self.assertRaisesRegex(ValueError,'Selection differs'):audit_development(self.source)

    def test_omitted_predefined_variant_is_rejected(self):
        self.results.pop();self.persist()
        with self.assertRaisesRegex(ValueError,'Missing predefined'):audit_development(self.source)

    def test_result_without_matching_raw_points_is_rejected(self):
        self.results[0]['metrics']['mean_net_points']=999.;self.persist()
        with self.assertRaisesRegex(ValueError,'summary differs'):audit_development(self.source)


if __name__=='__main__':unittest.main()
