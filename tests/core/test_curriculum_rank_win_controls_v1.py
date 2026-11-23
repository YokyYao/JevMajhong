import unittest
from tools.curriculum_rank_win_controls_v1 import BUCKETS,bucket,select,soft_target,METHOD


def row(identifier,split='train',kind='ron',rank=3,group=None):
    return dict(_meta=dict(id=identifier,split=split,category='win',group_id=group or identifier,
        placement=dict(rank=rank)),state=dict(win_context=dict(candidates=dict(win=dict(kind=kind)))),
        questions=dict(action=dict(criteria=dict(win=dict(type='hora'),skip=dict(type='none')))))


def score(income=1000,rank=2):return dict(supported=True,own_income=income,immediate_rank=rank)


class WinControlTests(unittest.TestCase):
    def test_cohorts_are_public_proxies_and_not_hard_accept_targets(self):
        r=row('a')
        self.assertEqual(bucket(r,score()),'sufficient_behind:ron')
        self.assertEqual(bucket(r,score(8000,3)),'valuable_still_behind:ron')
        self.assertIsNone(bucket(r,score(1000,3)))
        r['_meta']['placement']['rank']=2
        self.assertEqual(bucket(r,score()),'leading_accept_control:ron')
        self.assertIsNone(bucket(r,dict(supported=False)))

    def test_cohort_selection_preserves_splits_exclusions_and_family_caps(self):
        rows=[];scores={};quotas={p:{name:1 for name in BUCKETS} for p in ('train','validation')}
        for split in quotas:
            for index,name in enumerate(BUCKETS):
                phase,kind=name.split(':')
                for copy in range(2):
                    rid=f'{split}:{name}:{copy}'
                    r=row(rid,split,kind,2 if phase=='leading_accept_control' else 3,
                        group=f'{split}:{index}')
                    rows.append(r)
                    scores[rid]=score(8000 if phase=='valuable_still_behind' else 1000,
                        3 if phase=='valuable_still_behind' else 2)
        excluded={rows[0]['_meta']['id']}
        got=select(rows,scores,excluded,quotas)
        self.assertEqual(len(got),12)
        self.assertFalse(excluded & {r['_meta']['id'] for _,r in got})
        for split in quotas:
            family=[r['_meta']['group_id'] for _,r in got if r['_meta']['split']==split]
            self.assertEqual(len(set(family)),6)
        missing=list(rows[:-2])
        with self.assertRaises(ValueError):select(missing,scores,excluded,quotas)

    def test_complete_menu_target_includes_riichi_kan_and_prefers_win_on_placement_ties(self):
        keys=['hora','dahai','reach','ankan']
        def outcome(rank,net=0):return dict(full_match_completed=True,rank=rank,top2=rank<=2,net_from_anchor=net)
        worlds=[{k:outcome(2,1000 if k=='hora' else 20000) for k in keys} for _ in range(32)]
        result=soft_target(worlds,keys,2,1123)
        self.assertEqual(result['method'],METHOD);self.assertEqual(result['label'],'hora')
        self.assertTrue(all(p>0 for p in result['target'].values()))
        self.assertAlmostEqual(sum(result['target'].values()),1.)
        worlds=[{k:outcome(4 if k=='hora' else 2) for k in keys} for _ in range(32)]
        self.assertNotEqual(soft_target(worlds,keys,4,1123)['label'],'hora')

