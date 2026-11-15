import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from jevmahjong.core.io import digest
from jevmahjong.data.attack_mix import append_prepared, bootstrap_target
from tests.core.test_attack_states import attack_fixture


def record(opened, group, identifier):
    request = copy.deepcopy(attack_fixture(opened)['rows'][0]['public_request'])
    keys = list(request['questions']['action']['criteria'])
    request['questions']['action'].update(src='paired_future_masked',
        target={k: 1 / len(keys) for k in keys}, label=keys[0])
    request['_meta'] = dict(id=identifier, group_id=group, label_type='paired_future_masked',
        phase='open_discard' if opened else 'closed_discard')
    return request


def menu():
    request = record(False, 'native:1', 'new')
    rows = []
    for action in (1, 2):
        worlds = [dict(future_seed=i, future_wall_sha256=str(i),
            control=dict(mjai_log='zero'), branch=dict(mjai_log='win' if action == 1 else 'loss'))
            for i in range(3)]
        rows.append(dict(public_request=request, eligible=[0, 1, 2], teacher_action=0,
            action=action, episode_key=(1, 0, 0, action), hand_index=0, draws_before=0,
            counterfactual_worlds=worlds))
    return rows


def utility(log, *args):
    return dict(utility={'zero': 0., 'win': 1., 'loss': -1.}[log],
        own_win=float(log == 'win'), fast_reward=0., dealin=float(log == 'loss'))


class AttackMixTests(unittest.TestCase):
    def test_bootstrap_uses_shared_worlds_and_keeps_public_input_unchanged(self):
        rows = menu(); original = copy.deepcopy(rows)
        with patch('jevmahjong.data.attack_mix.hand_utility', utility):
            target = bootstrap_target(rows, dict(worlds=3, fast_gamma=.98, reward={}), 42, temperature=1.)
        expected = np.exp([0., 1., -1.]); expected /= expected.sum()
        np.testing.assert_allclose(list(target.values()), expected, atol=1e-14)
        self.assertEqual(rows, original)

    def test_bootstrap_rejects_missing_actions_and_unshared_futures(self):
        for mode in ('missing', 'duplicate', 'future', 'control', 'public'):
            rows = menu()
            if mode == 'missing': rows.pop()
            if mode == 'duplicate': rows.append(copy.deepcopy(rows[0]))
            if mode == 'future': rows[1]['counterfactual_worlds'][0]['future_seed'] = 99
            if mode == 'control': rows[1]['counterfactual_worlds'][0]['control']['mjai_log'] = 'win'
            if mode == 'public': rows[1]['public_request']['state']['honba'] = 4
            # The public request is shared in this synthetic menu; explicitly break the alias.
            if mode == 'public':
                rows = menu(); rows[1]['public_request'] = copy.deepcopy(rows[1]['public_request'])
                rows[1]['public_request']['state']['honba'] = 4
            with patch('jevmahjong.data.attack_mix.hand_utility', utility), self.assertRaises(ValueError):
                bootstrap_target(rows, dict(worlds=3, fast_gamma=.98, reward={}), 42)

    def test_bootstrap_retains_unknown_mask_and_nonzero_teacher_index(self):
        row = menu()[1]
        row['public_request']['questions']['action']['criteria']['a0']['sh'] = 2
        row.update(eligible=[1, 2], teacher_action=1)
        with patch('jevmahjong.data.attack_mix.hand_utility', utility):
            target = bootstrap_target([row], dict(worlds=3, fast_gamma=.98, reward={}), 42)
        self.assertEqual(target['a0'], 0.)
        self.assertGreater(target['a1'], target['a2'])
        self.assertGreater(target['a2'], 0.)

    def fixture(self, root):
        parent = dict(files={})
        for split in ('train', 'validation'):
            row = record(False, 'parent:' + split, 'parent:' + split)
            row['_meta']['label_type'] = 'expert_replay'
            path = root / (split + '.jsonl')
            path.write_bytes((json.dumps(row, separators=(',', ':')) + '\n').encode())
            parent['files'][split] = dict(path=str(path), sha256=digest(path), records=1)
        additions = [(s, record(opened, 'new:' + s, s + str(opened)))
            for s in ('train', 'validation') for opened in (False, True)]
        budget = {s: {p: 1 for p in ('closed_discard', 'open_discard')} for s in ('train', 'validation')}
        return parent, additions, budget

    def test_append_preserves_both_byte_prefixes_and_actual_group_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); parent, additions, budget = self.fixture(root); out = root / 'new'
            files, summary = append_prepared(parent, additions, out, budget, budget, budget)
            for split in budget:
                self.assertTrue((out / (split + '.jsonl')).read_bytes().startswith(
                    Path(parent['files'][split]['path']).read_bytes()))
                self.assertEqual(files[split]['records'], 3)
                self.assertEqual(summary[split]['open_discard']['walls'], 1)

    def test_append_rejects_duplicate_cross_split_unknown_and_quota_changes(self):
        for mode in ('duplicate', 'cross_split', 'blocked', 'unknown', 'count', 'threat'):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); parent, additions, budget = self.fixture(root); blocked = set()
                if mode == 'duplicate': additions.append(copy.deepcopy(additions[0]))
                if mode == 'cross_split': additions[2][1]['_meta']['group_id'] = 'new:train'
                if mode == 'blocked': blocked.add('new:train')
                if mode == 'unknown':
                    q = additions[0][1]['questions']['action']; keys = list(q['target'])
                    q['target'] = {keys[0]: .5, keys[1]: .5, keys[2]: 0.}
                if mode == 'count': budget['train']['closed_discard'] = 2
                if mode == 'threat': additions[0][1]['state']['riichi'] = [1]
                with self.assertRaises(ValueError):
                    append_prepared(parent, additions, root / 'new', budget, budget, budget, blocked)


if __name__ == '__main__': unittest.main()
