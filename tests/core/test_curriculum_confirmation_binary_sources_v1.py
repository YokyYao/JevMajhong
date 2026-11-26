"""Raw binary seed evidence must not silently skip values or model files."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
from tools.curriculum_confirmation_binary_sources_v1 import semantic_findings, scan_raw_rollout


class BinarySourceTests(unittest.TestCase):
    def test_nested_keys_values_strings_and_exact_numeric_boundaries(self):
        start=2026300000
        value={str(start):[start,start+31,start-1,start+32,'native:'+str(start+1)],
            'binary':str(start+2).encode(),'bool':True,'none':None,'cycle':[]}
        value['cycle'].append(value['cycle'])
        result=semantic_findings(value)
        self.assertEqual(result['reserved_occurrences'],5)

    def test_tensor_integer_float_complex_and_nonfinite_values_are_not_skipped(self):
        start=2026300000
        result=semantic_findings([torch.tensor([start,start+31,start+32],dtype=torch.int64),
            torch.tensor([float(start),float('nan')],dtype=torch.float64),
            torch.tensor([complex(start,0)],dtype=torch.complex128)])
        self.assertEqual(result['reserved_occurrences'],4)
        self.assertEqual(result['counts']['nonfinite_tensor_elements'],1)

    def test_unknown_objects_cannot_be_declared_clean(self):
        with self.assertRaisesRegex(ValueError,'Unclassified binary'): semantic_findings(object())

    def test_actual_torch_raw_rollout_schema_hash_and_future_fields_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.pt'
            torch.save({'rows':[{'episode_key':(12,0),'counterfactual_worlds':[{'future_seed':2026300000}]}]},path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest(); proof=scan_raw_rollout(path,digest)
            self.assertEqual(proof['reserved_occurrences'],1)
            self.assertEqual(proof['future_seed_fields'],1)
            with self.assertRaises(ValueError): scan_raw_rollout(path,'0'*64)

    def test_weight_files_and_untyped_future_seed_records_are_not_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.pt'
            for value in ({'weight':torch.zeros(2)}, {'rows':[{'counterfactual_worlds':[{'future_seed':'12'}]}]}):
                torch.save(value,path); digest=hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError): scan_raw_rollout(path,digest)


if __name__=='__main__': unittest.main()
