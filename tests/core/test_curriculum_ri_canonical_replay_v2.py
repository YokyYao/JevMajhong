import copy
import unittest
from tools.curriculum_ri_canonical_replay_v2 import canonical_replay, verify_metadata_only
from tests.core.test_curriculum_ri_replay_epoch_adapter_v1 import EpochAdapterTests


class CanonicalReplayTests(unittest.TestCase):
    def fixture(self):
        raw, plans, _, _ = EpochAdapterTests().fixture()
        loaded = copy.deepcopy(raw)
        for i, row in enumerate(loaded):
            row['_meta'].update(variant='clean', text_sha256=str(i), source='custom', row=i, split='custom')
        return raw, plans, loaded

    def test_loader_metadata_preserves_exact_admission_and_source_objects(self):
        raw, plans, loaded = self.fixture()
        before = copy.deepcopy(raw)
        replay = canonical_replay(raw, plans, 'TRAIN', lambda path: copy.deepcopy(loaded))
        n = len(raw)
        report = dict(requested_records=n, admitted_records=n, rejected_records=0, rejected=[])
        replay.admit(copy.deepcopy(loaded), report)
        self.assertEqual(raw, before)
        self.assertEqual(replay.rows, loaded)
        self.assertEqual(replay.canonicalization_report['added_metadata_counts']['text_sha256'], n)
        with self.assertRaises(ValueError): replay.admit(raw, report)
        with self.assertRaises(ValueError): replay.admit(loaded, dict(report, rejected_records=1))
        with self.assertRaises(ValueError): replay.admit(loaded[:-1], report)

    def test_state_label_target_and_existing_identity_changes_are_rejected(self):
        raw, plans, loaded = self.fixture()
        for mutate in (
                lambda row: row.update(state='changed'),
                lambda row: row['questions']['action'].update(label='changed'),
                lambda row: row['questions']['action'].update(target={'changed': 1}),
                lambda row: row['_meta'].update(id='changed'),
                lambda row: row['_meta'].update(group_id='changed')):
            changed = copy.deepcopy(loaded); mutate(changed[0])
            with self.assertRaises(ValueError): verify_metadata_only(raw, changed)

    def test_missing_reordered_or_unknown_metadata_is_rejected(self):
        raw, plans, loaded = self.fixture()
        with self.assertRaises(ValueError): verify_metadata_only(raw, loaded[:-1])
        with self.assertRaises(ValueError): verify_metadata_only(raw, list(reversed(loaded)))
        changed = copy.deepcopy(loaded); changed[0]['_meta']['unregistered'] = 'x'
        with self.assertRaises(ValueError): verify_metadata_only(raw, changed)
        changed = copy.deepcopy(loaded); del changed[0]['_meta']['id']
        with self.assertRaises(ValueError): verify_metadata_only(raw, changed)


if __name__ == '__main__': unittest.main()
