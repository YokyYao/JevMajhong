from collections import Counter
import json,os,socket,sys,unittest
from pathlib import Path
from tools.curriculum_targeted_win_source_v1 import wall_for

EXAMPLES={
 'win_reach':[0,0,0,1,2,3,12,13,14,24,25,26,28,28],
 'win_ankan':[0,0,0,0,1,2,12,13,14,24,25,26,28,28],
 'win_kakan':[0,0,0,0,1,2,12,13,14,31,31,31,28,28]}


class TargetWallTests(unittest.TestCase):
    def test_every_planned_physical_wall_partitions_all_tiles(self):
        for index,(kind,shape) in enumerate(EXAMPLES.items()):
            wall,plan=wall_for(kind,shape,2026610000+index)
            self.assertEqual(sorted(wall),list(range(136)))
            self.assertEqual(len(plan['own_initial']),13)
            self.assertNotIn(plan['first_trash']//4,shape)
            self.assertEqual(plan['win_tile']//4,shape[-1])

    def test_canonical_families_and_splits_are_frozen_before_native_values(self):
        from tools.curriculum_targeted_win_plans_v1 import plans
        config=dict(families_per_kind=dict(train=8,validation=4),seed_start=2026611000)
        result=plans(config)
        self.assertEqual(result,plans(config))
        self.assertEqual(Counter(p['split'] for p in result),dict(train=24,validation=12))
        self.assertEqual(len({p['canonical_shape_family'] for p in result}),36)


GPU=bool(os.environ.get('SLURM_JOB_ID') and (os.environ.get('SLURM_JOB_GPUS') or os.environ.get('SLURM_STEP_GPUS'))
    and socket.gethostname().startswith('ws-'))


@unittest.skipUnless(GPU,'Native conditional source tests require an allocated cluster GPU job')
class NativeTargetWinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        reg=json.loads((root/'runs/curriculum-progression-v1/fit-plan/registered.json').read_text())
        sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])

    def test_each_native_special_window_follows_real_turns_and_restores_exactly(self):
        from tools.curriculum_targeted_win_source_v1 import construct,restore
        from tools.curriculum_win_context import build_win_request
        from tools.curriculum_placement_full_win_policy_v1 import CompleteWinPlacementPolicyAgent
        from tools.curriculum_placement_input_v1 import PlacementPolicyClient
        from tools.verify_curriculum_rank_win_input_v1 import Selector
        from jevmahjong.core.state import signature
        for index,(kind,shape) in enumerate(EXAMPLES.items()):
            env,obs,source=construct(kind,shape,2026610000+index,[33000,25000,22000,20000])
            request,actions,_=build_win_request(obs,env.mjai_log[-1])
            self.assertEqual(request['state']['hand_number'],4)
            self.assertFalse(env.is_first_turn)
            self.assertGreaterEqual(len(source['trace']),5)
            row=dict(request,_meta=dict(native_source=source))
            fork,other=restore(row)
            self.assertEqual(build_win_request(other,fork.mjai_log[-1])[0],request)
            criteria=request['questions']['action']['criteria'];keys=list(criteria)
            expected={'win_reach':'reach','win_ankan':'ankan','win_kakan':'kakan'}[kind]
            key=next(k for k,v in criteria.items() if v['type']==expected)
            model=Selector(key);agent=CompleteWinPlacementPolicyAgent(PlacementPolicyClient(model))
            got=agent.act(obs,current_public_event=env.mjai_log[-1])
            self.assertEqual(model.calls,1)
            self.assertEqual(signature(got),signature(actions[keys.index(key)]))

