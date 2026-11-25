import copy
import unittest
from types import SimpleNamespace

from tools import curriculum_ri_parent_anchor_v1 as anchor
from tools import curriculum_ri_inverse_repeat_loss_v1 as inverse
from tools import curriculum_yaku_parent_loss_v1 as yaku
from tools import curriculum_yaku_ri_coverage_v1 as schedule
from tools import curriculum_yaku_ri_anchor_loss_v1 as objective
from tools import curriculum_yaku_ri_anchor_epoch_v1 as epoch
from tests.core.test_curriculum_ri_parent_anchor_v1 import fixture


class YakuAnchorEpochTests(unittest.TestCase):
    def population(self):
        rows = []; predictions = []; categories = []
        for source in yaku.MASS:
            count = 25 if source in anchor.RI_SOURCES and ':teacher:' in source else 4
            for _ in range(count):
                index = len(rows)
                kinds = ['dahai', 'reach', 'dahai'] if source in anchor.RI_SOURCES else ['dahai'] * 3
                label = 1 if source.endswith(':riichi_yes') else 2
                target = [.3, 0., .7] if ':mc:' in source else None
                row, _ = fixture(source, kinds, label, str(index), target)
                row['state'] = dict(public_test_index=index)
                rows.append(row); categories.append(source)
                if source in anchor.RI_SOURCES:
                    predictions.append(dict(id=str(index), keys=['k0', 'k1', 'k2'], p=[.15, .6, .25]))
        plans = [schedule.build(categories, list(range(len(rows))), seed, batch=11) for seed in (31, 32)]
        return rows, plans, objective.make_targets(rows, predictions)

    def args(self):
        return SimpleNamespace(batch=1, accum=11, rlcr=False, p_none=0.,
            p_none_distract=0., p_distract=0., p_none_pair=0.)

    def encoder(self, model, tok, args, rows, number):
        result = []
        for row in rows:
            stripped = {key: copy.deepcopy(value) for key, value in row.items() if key != '_meta'}
            raw = stripped['questions']['action']; keys = list(raw['criteria'])
            question = dict(qid='action', src=raw['src'], keys=keys, label=keys.index(raw['label']),
                target=[raw['target'].get(key, 0.) for key in keys] if raw.get('target') else None)
            record = yaku.annotate(dict(questions=[question]), stripped)
            self.assertNotIn(inverse.FIELD, question)
            self.assertNotIn(anchor.FIELD, question)
            result.append(SimpleNamespace(rec=record,
                enc=dict(ids=[row['state']['public_test_index'], 9])))
        return result

    def test_both_full_epochs_resolve_exact_originals_with_metadata_stripped_encoder(self):
        rows, plans, targets = self.population(); before = copy.deepcopy(rows)
        replay = epoch.YakuAnchoredEpochReplay(rows, plans, targets)
        for number in (0, 1):
            replay.shuffle(list(range(replay.size)))
            for position, virtual in enumerate(replay.virtual_requests):
                variant = replay.encode_batch(self.encoder, None, None, self.args(), [virtual], number)[0]
                index = plans[number]['indices'][position]; question = variant.rec['questions'][0]
                self.assertEqual(variant.enc, dict(ids=[index, 9]))
                self.assertNotIn('_meta', variant.rec)
                self.assertEqual(question[inverse.MARKER], objective.WEIGHT_METHOD)
                self.assertEqual(question[anchor.FIELD] is not None, question['src'] in anchor.RI_SOURCES)
        receipt = replay.finish()
        self.assertEqual(receipt['method'], epoch.METHOD)
        self.assertEqual(receipt['unique_rows_each_epoch'], [len(rows), len(rows)])
        self.assertEqual(rows, before)

    def test_missing_targets_and_repeated_attachment_are_rejected(self):
        rows, plans, targets = self.population(); incomplete = dict(targets); incomplete.pop(next(iter(incomplete)))
        with self.assertRaises(ValueError):
            epoch.YakuAnchoredEpochReplay(rows, plans, incomplete)
        replay = epoch.YakuAnchoredEpochReplay(rows, plans, targets)
        replay.shuffle(list(range(replay.size)))
        rec = replay.encode_batch(self.encoder, None, None, self.args(), [replay.virtual_requests[0]], 0)[0].rec
        original = rows[plans[0]['indices'][0]]
        with self.assertRaises(ValueError):
            epoch.attach_targets(rec, original, targets)

    def test_old19_epoch_module_remains_bound_to_its_original_mass_table(self):
        from tools import curriculum_ri_replay_epoch_adapter_v1 as old
        before = copy.deepcopy(old.MASS)
        rows, plans, targets = self.population()
        epoch.YakuAnchoredEpochReplay(rows, plans, targets)
        self.assertEqual(old.MASS, before)
        self.assertEqual(len(old.MASS), 19)


if __name__ == '__main__':
    unittest.main()
