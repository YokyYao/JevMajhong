import copy
import json
from pathlib import Path
import tempfile
import unittest

from jevmahjong.core.io import digest
from jevmahjong.data.served_targets import POLICY, conditional_target, prepare


def record():
    return dict(state=dict(seat=0, hand=['1m'], riichi=[1],
        melds=[[dict(type=1)], [], [], []], rivers=[[], [], [], []]),
        questions=dict(action=dict(criteria=dict(
            a0=dict(type='dahai', pai='1m', sh=0, ukeire=5, red=False, safe=[]),
            a1=dict(type='dahai', pai='9m', sh=1, ukeire=10, red=False, safe=[1]),
            a2=dict(type='dahai', pai='3p', sh=0, ukeire=2, red=False, safe=[])),
            target=dict(a0=.1, a1=.8, a2=.1), label='a1')),
        _meta=dict(id='mc', group_id='native:1', phase='open_discard', label_type='paired_future_masked'))


class ServedTargetsTests(unittest.TestCase):
    def test_removes_only_excluded_mass_without_mutating_public_input(self):
        row = record(); original = copy.deepcopy(row)
        target, stats = conditional_target(row)
        self.assertEqual(target, dict(a0=.5, a1=0., a2=.5))
        self.assertEqual(stats, dict(excluded_mass=.8, preferred_excluded=True))
        self.assertEqual(row, original)

    def test_rejects_unknown_served_action_or_invalid_target(self):
        for target in [dict(a0=0., a1=.8, a2=.2), dict(a0=.1, a1=float('nan'), a2=.1),
                dict(a0=.1, a1=.9, a2=.1), dict(a0=.1, a1=-.1, a2=1.)]:
            row = record(); row['questions']['action']['target'] = target
            with self.assertRaises(ValueError): conditional_target(row)

    def test_rejects_expert_closed_and_nonordinary_inputs(self):
        for kind in ('expert', 'phase', 'closed', 'kan'):
            row = record()
            if kind == 'expert': row['_meta']['label_type'] = 'expert_replay'
            if kind == 'phase': row['_meta']['phase'] = 'closed_discard'
            if kind == 'closed': row['state']['melds'][0] = []
            if kind == 'kan': row['questions']['action']['criteria']['a1']['type'] = 'ankan'
            with self.assertRaises(ValueError): conditional_target(row)

    def test_dataset_changes_only_open_mc_targets_and_preserves_validation_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); mc = record(); expert = copy.deepcopy(mc)
            expert['_meta'].update(id='expert', group_id='expert:1', label_type='expert_replay')
            other = copy.deepcopy(mc); other['_meta'].update(id='other', group_id='native:2', phase='closed_discard')
            original = [json.dumps(r, separators=(',', ':')).encode() + b'\n' for r in [expert, mc, other]]
            train = root / 'train.jsonl'; train.write_bytes(b''.join(original))
            validation = root / 'validation.jsonl'; validation.write_bytes(b'fixed validation bytes\n')
            receipt = root / 'result.json'
            receipt.write_text(json.dumps(dict(status='balanced_raw_data_ready', requires_masked_soft_target_loss=True,
                files={s:dict(path=str(p), sha256=digest(p), records=n) for s,p,n in
                    [('train', train, 3), ('validation', validation, 1)]})))
            out = root / 'conditioned'
            result = prepare(dict(policy=POLICY, parent_data_receipt=str(receipt),
                parent_data_receipt_sha256=digest(receipt), out=str(out)))
            lines = (out / 'train.jsonl').read_bytes().splitlines(keepends=True)
            self.assertEqual(lines[0], original[0]); self.assertEqual(lines[2], original[2])
            changed = json.loads(lines[1]); q = changed['questions']['action']
            self.assertEqual(q['target'], dict(a0=.5, a1=0., a2=.5)); self.assertEqual(q['label'], 'a0')
            q.update(target=mc['questions']['action']['target'], label='a1')
            self.assertEqual(changed, mc)
            self.assertEqual((out / 'validation.jsonl').read_bytes(), validation.read_bytes())
            self.assertEqual(result['support_conditioning']['preferred_excluded'], 1)


if __name__ == '__main__': unittest.main()
