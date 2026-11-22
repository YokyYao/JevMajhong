from collections import Counter
import unittest
from tools.curriculum_public_inventory import inventory
from tools.curriculum_public_worlds import SamplingExhausted
from tools.curriculum_hierarchical_prior import sample


class HierarchicalPriorTests(unittest.TestCase):
    def pool(self,required):
        return inventory(0,list(range(13)),[[],[],[],[]],[[],[],[],[]],[135],[p in required for p in range(4)],'ron')
    def history(self,required):
        return [dict(type='start_kyoku'),*[dict(type='reach_accepted',actor=p) for p in required],dict(type='tsumo',actor=0)]

    def test_three_riichi_inventory_partition_and_seed_are_exact(self):
        pool=self.pool([1,2,3]);history=self.history([1,2,3]);riichi=[False,True,True,True]
        waits=lambda p,h:[1] if sum(h)%2==0 else []
        a=sample(pool,history,riichi,17,3,100,waits)
        self.assertEqual(a,sample(pool,history,riichi,17,3,100,waits))
        self.assertFalse(a['uniform_joint_target_for_multi_riichi'])
        self.assertFalse(a['uniform_joint_equivalent_when_opponent_riichi_le1'])
        for world in a['allocations']:
            all_tiles=[*world['unknown_nonhand_tiles'],*[t for h in world['opponent_hands'].values() for t in h]]
            self.assertEqual(Counter(all_tiles),Counter(pool['unseen_tiles']))
            self.assertTrue(all(sum(h)%2==0 for h in world['opponent_hands'].values()))
            self.assertEqual(set(world['proposal_riichi_order']),{1,2,3})

    def test_zero_and_single_riichi_declare_the_correct_overlap_scope(self):
        for required in ([],[1]):
            got=sample(self.pool(required),self.history(required),[p in required for p in range(4)],8,1,10,
                lambda p,h:[1] if p in required else [])
            self.assertTrue(got['uniform_joint_equivalent_when_opponent_riichi_le1'])
            self.assertFalse(got['training_labels'])

    def test_history_incompatible_worlds_exhaust_without_partial_return(self):
        history=[dict(type='start_kyoku'),dict(type='reach_accepted',actor=1),dict(type='dahai',actor=2,tile_type=0),dict(type='tsumo',actor=0)]
        with self.assertRaises(SamplingExhausted) as caught:
            sample(self.pool([1]),history,[False,True,False,False],5,1,4,lambda p,h:[0] if p==1 else [])
        self.assertEqual(caught.exception.attempts,4);self.assertEqual(caught.exception.accepted,0)

    def test_cannot_silently_drop_a_public_riichi_constraint(self):
        with self.assertRaises(ValueError):sample(self.pool([1]),self.history([1]),[False]*4,1,1,5,lambda p,h:[])


if __name__=='__main__':unittest.main()
