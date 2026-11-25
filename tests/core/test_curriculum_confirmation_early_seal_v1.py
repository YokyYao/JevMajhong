"""Reject incomplete/corrupt final evidence without opening reserved games."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import seal_curriculum_confirmation_early_v1 as seal


class FinalSealTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def save(self, name, value):
        path = self.root/name
        path.write_text(json.dumps(value)+'\n')
        return path

    def source_fixture(self):
        producer = self.save('producer.json', {'fixture': True})
        artifact = self.save('artifact.json', {'seed': 1})
        proof = dict(status=seal.SOURCE_STATUS,
            candidate_identity='candidate-fixture', reference_identity='reference-fixture',
            seed_start=2026300000, seed_groups=32, reserved_source_data_hits=0,
            unreviewed_artifacts=[], preliminary_probe_used_as_final_proof=False,
            reserved_games_generated=0, producer_source_files={str(producer): seal.sha(producer)},
            artifact_files={str(artifact): seal.sha(artifact)},
            json_artifact_classifications=[{'fixture_only': True}])
        path = self.save('proof.json', proof)
        self.save('completion-binding-private.json', dict(files={
            str(p): seal.sha(p) for p in (producer, artifact, path)}))
        return path, proof, artifact

    def require_source(self, path, files=None):
        return seal.require_source_receipt(path,
            {'identity_sha256': 'candidate-fixture'}, {'identity_sha256': 'reference-fixture'},
            {} if files is None else files)

    def test_criteria_file_is_exact_original_predeclared_criteria(self):
        seal.require_criteria(seal.read(seal.CRITERIA_PATH))
        changed = copy.deepcopy(seal.CRITERIA); changed['seed_groups'] = 2
        with self.assertRaises(ValueError): seal.require_criteria(changed)

    def test_snapshot_closure_cannot_supply_final_provenance(self):
        path = self.save('snapshot.json', dict(
            status='snapshot250_actual_student_source_lineages_and_families_closed_not_final_candidate'))
        with self.assertRaises(ValueError): self.require_source(path)

    def test_mutated_artifact_rejects_even_when_summary_claims_clean(self):
        path, _, artifact = self.source_fixture()
        artifact.write_text('{"seed":2026300000}\n')
        with self.assertRaises(ValueError): self.require_source(path)

    def test_missing_original_producer_receipt_rejects(self):
        path, _, _ = self.source_fixture()
        (self.root/'completion-binding-private.json').unlink()
        with self.assertRaises(FileNotFoundError): self.require_source(path)

    def test_receipt_must_bind_the_proof_not_only_its_artifacts(self):
        path, _, _ = self.source_fixture()
        receipt = seal.read(self.root/'completion-binding-private.json')
        del receipt['files'][str(path)]
        self.save('completion-binding-private.json', receipt)
        with self.assertRaises(ValueError): self.require_source(path)

    def test_proof_for_different_actual_candidate_rejects(self):
        path, proof, _ = self.source_fixture()
        proof['candidate_identity'] = 'another-candidate'
        self.save('proof.json', proof)
        with self.assertRaises(ValueError): self.require_source(path)

    def test_reserved_or_incomplete_anchor_rejects(self):
        anchor = dict(seed=2026300000, profile='efficiency', seat=0,
            completed=True, ledger_passed=True, mjai_log=[{'type': 'start_game'}])
        path = self.save('matches-private.jsonl', anchor)
        with self.assertRaises(ValueError): seal.require_anchor(path)
        anchor.update(seed=2026200000, completed=False)
        self.save('matches-private.jsonl', anchor)
        with self.assertRaises(ValueError): seal.require_anchor(path)

    def test_existing_partial_seal_is_preserved_before_any_candidate_work(self):
        partial = self.root/'final'; partial.mkdir()
        marker = partial/'partial.json'; marker.write_text('preserve\n')
        with patch.object(seal, 'OUT', partial), patch.object(seal, 'resource'), \
                patch.object(seal, 'make_registration') as make:
            with self.assertRaises(ValueError): seal.seal(self.root/'missing', self.root/'missing')
        make.assert_not_called()
        self.assertEqual(marker.read_text(), 'preserve\n')

    def test_missing_source_prerequisite_leaves_no_actual_seal(self):
        output = self.root/'not-created'
        with patch.object(seal, 'OUT', output), patch.object(seal, 'resource'), \
                patch.object(seal, 'make_registration', side_effect=FileNotFoundError('actual stage4 incomplete')):
            with self.assertRaises(FileNotFoundError): seal.seal(self.root/'missing', self.root/'missing')
        self.assertFalse(output.exists())

    def test_conflicting_hashes_cannot_be_silently_replaced(self):
        artifact = self.save('artifact.json', {'fixture': True})
        with self.assertRaises(ValueError): seal.add_file({str(artifact): 'different'}, artifact)


if __name__ == '__main__': unittest.main()
