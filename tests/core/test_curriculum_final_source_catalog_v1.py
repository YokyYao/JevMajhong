import json
from pathlib import Path
import tempfile
import unittest

from tools.prove_curriculum_final_source_catalog_v1 import (
    add_role, close_graph, merge_identities, require_lineage_coverage)
from tools.inventory_curriculum_source_delta_v1 import Inventory
from tools.curriculum_transfer import sha

HISTORY = {'original_manifest': ''}


class FinalSourceCatalogueBoundaries(unittest.TestCase):
    def test_a_later_receipt_cannot_replace_a_completed_identity(self):
        destination = {'/tmp/a': 'a'*64}
        merge_identities(destination, {'/tmp/a': 'a'*64})
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            merge_identities(destination, {'/tmp/a': 'b'*64})

    def test_a_later_role_cannot_hide_a_different_source(self):
        catalogue = {}
        add_role(catalogue, '/tmp/data.json', 'a'*64, 'closed_data', 'receipt')
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            add_role(catalogue, '/tmp/data.json', 'b'*64, 'parameter', 'another_receipt')

    def test_ancestral_validation_and_train_both_need_typed_coverage(self):
        protocol = dict(trainer_args=dict(eval_data='/tmp/validation.jsonl'),
            input_sha256={'/tmp/train.jsonl':'a'*64, '/tmp/validation.jsonl':'b'*64})
        catalogue = {'/tmp/train.jsonl':dict(sha256='a'*64)}
        with self.assertRaisesRegex(ValueError, 'TRAIN/validation'):
            require_lineage_coverage({'candidate':protocol}, {'candidate':['/tmp/train.jsonl']}, catalogue)
        catalogue['/tmp/validation.jsonl'] = dict(sha256='b'*64)
        require_lineage_coverage({'candidate':protocol}, {'candidate':['/tmp/train.jsonl']}, catalogue)

    def test_new_jsonl_is_not_silently_classified_by_its_name(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'labels.jsonl'; path.write_text('{"label":1}\n')
            graph=Inventory({}, HISTORY, root=folder); graph.add(str(path),sha(path),str(path),'/root')
            with self.assertRaisesRegex(ValueError, 'typed source evidence'):
                close_graph(graph, {}, set(), {})

    def test_new_metadata_cannot_hide_a_reserved_exponent_value(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'metadata.json'; path.write_text('{"seed":2.026300001e9}')
            graph=Inventory({}, HISTORY, root=folder); graph.add(str(path),sha(path),str(path),'/root')
            with self.assertRaisesRegex(ValueError, 'seed context'):
                close_graph(graph, {}, set(), {})

    def test_previously_typed_case_still_needs_new_hash_edge_closure(self):
        with tempfile.TemporaryDirectory() as folder:
            hidden=Path(folder)/'unobserved.pt'; hidden.write_bytes(b'not a typed source')
            case=Path(folder)/'case.json'; case.write_text(json.dumps({'files':{str(hidden):sha(hidden)}}))
            catalogue={str(case):dict(sha256=sha(case))}
            graph=Inventory({str(case):sha(case)}, HISTORY, root=folder)
            graph.add(str(case),sha(case),str(case),'/root')
            with self.assertRaisesRegex(ValueError, 'typed source evidence'):
                close_graph(graph, catalogue, {str(case)}, {})

    def test_old_closed_payloads_are_not_decoded_again(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'closed.pt'; path.write_bytes(b'closed original raw artifact')
            catalogue={str(path):dict(sha256=sha(path))}
            graph=Inventory({str(path):sha(path)}, HISTORY, root=folder)
            graph.add(str(path),sha(path),str(path),'/root')
            self.assertEqual(close_graph(graph,catalogue,set(),{}),([],0))

    def test_metadata_changed_after_declaration_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'metadata.json'; path.write_text('{"step":2500}')
            graph=Inventory({}, HISTORY, root=folder); graph.add(str(path),sha(path),str(path),'/root')
            path.write_text('{"step":2501}')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                close_graph(graph,{},set(),{})


if __name__ == '__main__': unittest.main()
