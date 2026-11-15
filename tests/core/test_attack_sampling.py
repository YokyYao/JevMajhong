import copy
import hashlib
import unittest
from unittest.mock import patch

from jevmahjong.training.ev_collect import select_case_plan


class AttackSamplingTests(unittest.TestCase):
    def fixture(self):
        parts={split:[dict(episode_key=(wall,0,turn),request_sha256=str(wall)+'/'+str(turn),
            training_advantage=float(turn)) for wall in walls for turn in range(8)]
            for split,walls in [('train',range(4)),('held',range(4,8))]}
        cfg=dict(train_scopes=['ordinary'],decision_filter='attacking_closed_discard',
            seed=11,canonical_state_only=True,max_states=8,max_states_by_split=dict(train=4,held=4),
            whole_wall_state_cap_by_split=dict(train=1,held=1),public_state_order='seeded_public_hash_v1')
        return parts,cfg

    def select(self,parts,cfg):
        def choose(values,*args):return [copy.deepcopy(row) for rows in values.values() for row in rows]
        with patch('jevmahjong.training.ev_collect.choose_cases',choose):
            return select_case_plan(parts,None,cfg)

    def test_hash_order_preserves_wall_budgets_and_uses_only_public_origin(self):
        parts,cfg=self.fixture();actual=self.select(parts,cfg)
        self.assertEqual(len(actual),8);self.assertEqual(len({r['episode_key'][0] for r in actual}),8)
        for split,rows in parts.items():
            for wall in {r['episode_key'][0] for r in rows}:
                candidates=[r for r in rows if r['episode_key'][0]==wall]
                expected=min(candidates,key=lambda r:hashlib.sha256(
                    (str(cfg['seed'])+'|'+split+'|'+str(wall)+'|0|'+r['request_sha256']).encode()).hexdigest())
                self.assertIn(expected['episode_key'],[r['episode_key'] for r in actual])
        self.assertTrue(any(r['episode_key'][2]>0 for r in actual))
        changed=copy.deepcopy(parts)
        for rows in changed.values():
            rows.reverse()
            for row in rows:row['training_advantage']=-999*row['training_advantage']
        self.assertEqual([r['episode_key'] for r in actual],
            [r['episode_key'] for r in self.select(changed,cfg)])

    def test_legacy_order_is_unchanged_without_the_explicit_new_option(self):
        parts,cfg=self.fixture();cfg.pop('public_state_order')
        self.assertEqual([r['episode_key'][2] for r in self.select(parts,cfg)],[0]*8)

    def test_rejects_undeclared_scope_or_missing_budget(self):
        parts,cfg=self.fixture()
        for change in [dict(public_state_order='random'),dict(decision_filter='kan_windows'),
                dict(canonical_state_only=False),dict(whole_wall_state_cap_by_split=None),
                dict(max_states_by_split=None)]:
            with self.assertRaisesRegex(ValueError,'Public hash ordering'):
                self.select(parts,dict(cfg,**change))


if __name__=='__main__':unittest.main()
