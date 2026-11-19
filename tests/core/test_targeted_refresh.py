from collections import Counter
import copy
import json
from pathlib import Path
import unittest

from jevmahjong.endgame.targeted_refresh import reliable_target,world_design,select_manifest,seeded_future,balanced_batches
from tools.build_targeted_endgame import validate_config


class TargetedRefreshTests(unittest.TestCase):
    def test_world_blocks_balance_policies_and_opponents_independently(self):
        for start in (0,32):
            counts=Counter((world_design(i)['policy'],world_design(i)['opponents']) for i in range(start,start+32))
            self.assertEqual(set(counts.values()),{8});self.assertEqual(len(counts),4)
        with self.assertRaises(ValueError):world_design(64)

    def test_rare_uncertain_success_is_kept_as_soft_evidence(self):
        prior={'pass':.8,'pon':.2};values={'pass':[0]*64,'pon':[1]+[0]*63}
        target,report=reliable_target(prior,values,list(prior))
        self.assertGreater(target['pon'],0);self.assertLess(target['pon'],1)
        self.assertTrue(report['rare_successes_kept'])
        self.assertGreater(report['paired']['pon']['paired_advantage'],0)
        self.assertEqual(report['paired']['pon']['shrunk_advantage'],0)

    def test_negative_and_replication_disagreement_are_not_discarded(self):
        prior={'a':.7,'b':.3};values={'a':[1]*32+[0]*32,'b':[0]*32+[1]*16+[0]*16}
        target,report=reliable_target(prior,values,list(prior))
        self.assertTrue(report['paired']['b']['negative_evidence_retained'])
        self.assertEqual(report['paired']['b']['block_advantages'],[-1.,.5])
        self.assertGreaterEqual(sum(target[k]*sum(values[k])/64 for k in prior),
                                sum(prior[k]*sum(values[k])/64 for k in prior)-1e-12)

    def test_incomplete_or_unpaired_worlds_are_rejected(self):
        with self.assertRaises(ValueError):reliable_target({'a':1.},{'a':[0]*32},['a'])
        with self.assertRaises(ValueError):reliable_target({'a':1.},{'a':[2]*64},['a'])

    def test_source_selection_is_outcome_blind_and_one_per_game(self):
        def row(seed,kind,cell,key):return dict(seed=seed,id=str(seed)+kind,split='train',category=kind,cell=cell,selection_key=key)
        rows=[row(1,'chi','near','a'),row(1,'discard','near','a'),row(2,'discard','far','b'),row(3,'discard','near','c')]
        selected=select_manifest(rows,{'train':{'chi':1,'discard':2}})
        self.assertEqual(len({r['seed'] for r in selected}),3)
        changed=copy.deepcopy(rows)
        for r in changed:r['hypothetical_future_score']=999999
        self.assertEqual([r['id'] for r in selected],[r['id'] for r in select_manifest(changed,{'train':{'chi':1,'discard':2}})])
        with self.assertRaises(ValueError):select_manifest(rows,{'train':{'chi':2}})

    def test_source_mass_and_batch_quota_are_normalized_not_row_duplication(self):
        c=json.loads((Path(__file__).resolve().parents[2]/'configs/training/endgame-targeted-refresh-v1.json').read_text())
        validate_config(c);self.assertAlmostEqual(sum(c['loss_mass'].values()),1.)
        self.assertAlmostEqual(sum(v for k,v in c['loss_mass'].items() if k.startswith('new_')),.4)
        rows=[{'questions':{'action':{'src':k}}} for k in c['batch_quotas'] for _ in range(10)]
        for batch in balanced_batches(rows,c['batch_quotas'],5):
            self.assertEqual(dict(Counter(rows[i]['questions']['action']['src'] for i in batch)),c['batch_quotas'])

    def test_future_seed_varies_next_hand_without_changing_current_public_state(self):
        from riichienv import RiichiEnv,GameRule
        from jevmahjong.core.seeds import engine_seed
        from jevmahjong.evaluation.agents import EfficiencyAgent
        from jevmahjong.training.collector import call_observation_digest
        env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(42,'spaced-v1'));env.reset()
        a=seeded_future(env,12345,100);b=seeded_future(env,12345,200)
        self.assertEqual(call_observation_digest(a.clone().get_observations()),call_observation_digest(b.clone().get_observations()))
        original=(a.round_wind,a.kyoku_idx,a.honba);agent=EfficiencyAgent()
        for _ in range(1000):
            if (a.round_wind,a.kyoku_idx,a.honba)!=original:break
            for e in (a,b):
                obs=e.get_observations();e.step({p:agent.act(o) for p,o in obs.items() if o.legal_actions()})
        else:self.fail('Bounded first hand did not complete')
        self.assertFalse(a.done());self.assertFalse(b.done())
        self.assertEqual(list(a.scores()),list(b.scores()));self.assertNotEqual(a.hands,b.hands)


if __name__=='__main__':unittest.main()
