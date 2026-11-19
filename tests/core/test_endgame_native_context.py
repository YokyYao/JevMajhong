import copy
import unittest
from riichienv import GameRule,RiichiEnv
from jevmahjong.core.seeds import engine_seed
from jevmahjong.endgame.collect import native_match
from jevmahjong.endgame.native_context import replay_decisions,select_context_rows,phase_for
from tools.prepare_native_context_replay import allocation_metadata


class NativeContextTests(unittest.TestCase):
    def test_companion_cpu_step_must_have_a_real_parent_gpu_allocation(self):
        description='JobId=123 JobState=RUNNING NodeList=ws-l3-011 AllocTRES=cpu=4,mem=40G,gres/gpu=1'
        self.assertEqual(allocation_metadata(description,'ws-l3-011')['JobId'],'123')
        for bad in (description.replace('gres/gpu=1','gres/gpu=0'),
                    description.replace('JobState=RUNNING','JobState=COMPLETED'),
                    description.replace('ws-l3-011','lo-01')):
            with self.assertRaises(RuntimeError):allocation_metadata(bad,'ws-l3-011')

    def test_exact_event_replay_and_initialization_mismatch(self):
        source=native_match(2026111900)
        env=RiichiEnv(game_mode='4p-red-east',rule=GameRule.default_mjsoul(),seed=engine_seed(2026111900,'spaced-v1'))
        positions=list(replay_decisions(env,source['mjai_log']))
        self.assertTrue(positions)
        self.assertEqual(env.mjai_log,source['mjai_log'])
        self.assertEqual(list(env.scores()),source['final_scores'])
        broken=copy.deepcopy(source['mjai_log']);broken[0]['type']='invalid'
        with self.assertRaises(ValueError):list(replay_decisions(env,broken))

    def test_rank_quotas_do_not_greedily_consume_critical_sources(self):
        rows=[]
        for split,n,start in (('train',32,0),('validation',8,100)):
            # One half can only supply rank3; shared half can supply either.
            # A sampler that consumes shared groups for rank3 first will fail.
            for offset in range(2*n):
                for rank in ((3,) if offset<n else (3,4)):
                    rows.append({'seed':start+offset,'rank':rank,'split':split,
                        'cell':f'{rank}:discard','selection_key':f'{2*n-offset:04d}'})
        chosen=select_context_rows(rows)
        self.assertEqual(len(chosen),80)
        self.assertEqual(len({r['seed'] for r in chosen}),80)
        for split,n in (('train',32),('validation',8)):
            self.assertEqual(sum(r['split']==split and r['rank']==3 for r in chosen),n)
            self.assertEqual(sum(r['split']==split and r['rank']==4 for r in chosen),n)
        changed=copy.deepcopy(rows)
        for row in changed:row['future_rank']=1
        self.assertEqual([(r['seed'],r['rank']) for r in chosen],[(r['seed'],r['rank']) for r in select_context_rows(changed)])

    def test_split_leakage_and_incomplete_coverage_are_not_silently_relaxed(self):
        with self.assertRaises(ValueError):select_context_rows([])
        with self.assertRaises(ValueError):select_context_rows([
            {'seed':1,'split':'train'},{'seed':1,'split':'validation'}])

    def test_decision_phase_preserves_all_three_kan_types(self):
        for kind in ('ankan','kakan','daiminkan'):
            self.assertEqual(phase_for({kind,'none'}),'kan')
        self.assertEqual(phase_for({'pon','chi','none'}),'call')
        self.assertEqual(phase_for({'reach','dahai'}),'riichi')
        self.assertEqual(phase_for({'dahai'}),'discard')


if __name__=='__main__':unittest.main()
