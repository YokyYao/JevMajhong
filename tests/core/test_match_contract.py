import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from jevmahjong.evaluation.match_contract import calling_comparison_contract,check_evaluator_package


class MatchContractTests(unittest.TestCase):
    def test_stronger_reference_is_bound_for_context_and_head_without_double_application(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'baseline.pt';path.write_bytes(b'fixed-baseline')
            sha=hashlib.sha256(path.read_bytes()).hexdigest();spec={'checkpoint':str(path),'sha256':sha}
            cfg={'confirmation_objective':'hand_win','updated_scopes':['calling'],
                'candidate_activation':'call_windows','call_context_baseline':spec}
            proto={'config':cfg,'model_files':{},'call_context_baseline':spec}
            fit={'config':{'train_scopes':['calling']},'model_files':{},'call_context_baseline':spec}
            model={'base_and_LoRA_frozen':True,'trained_scopes':['calling'],'model_files':{},
                'call_context_baseline':spec,'decision_filter':'call_windows','kan_context':{},
                'context_replaces_calling_baseline':True,'context_initialization':'calling_baseline'}
            contract=calling_comparison_contract(cfg,proto,fit,model,sha,'call_windows')
            self.assertEqual(contract['candidate_component'],'calling_context')
            self.assertEqual(contract['expected_native_parity']['ordinary_kan'],2)
            for change in ({'context_replaces_calling_baseline':False},{'context_initialization':'zero_output'},
                           {'call_context_baseline':None}):
                with self.subTest(change=change),self.assertRaises(ValueError):
                    calling_comparison_contract(cfg,proto,fit,dict(model,**change),sha,'call_windows')
            with self.assertRaisesRegex(ValueError,'Prescribed reference policy'):
                calling_comparison_contract(cfg,proto,fit,model,'0'*64,'call_windows')
            head=dict(model);head.pop('kan_context');head.update(decision_filter='all',context_replaces_calling_baseline=False)
            cfg=dict(cfg,candidate_activation='all');proto=dict(proto,config=cfg)
            contract=calling_comparison_contract(cfg,proto,fit,head,sha,'all')
            self.assertEqual(contract['candidate_component'],'calling_pointer_head')
            self.assertNotIn('ordinary_kan',contract['expected_native_parity'])
            with self.assertRaises(ValueError):
                calling_comparison_contract(cfg,proto,fit,dict(head,context_replaces_calling_baseline=True),sha,'all')
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Reference checkpoint changed'):
                calling_comparison_contract(cfg,proto,fit,head,sha,'all')

    def test_legacy_reference_and_frozen_evaluator_source_are_explicit(self):
        cfg={'confirmation_objective':'hand_win','updated_scopes':['calling'],'candidate_activation':'call_windows'}
        proto={'config':cfg,'model_files':{}}
        fit={'config':{'train_scopes':['calling']},'model_files':{}}
        model={'base_and_LoRA_frozen':True,'trained_scopes':['calling'],'model_files':{},
            'decision_filter':'call_windows','kan_context':{}}
        self.assertEqual(calling_comparison_contract(cfg,proto,fit,model,None,'call_windows')['candidate_component'],'calling_context')
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary)/'source';package=source/'jevmahjong';package.mkdir(parents=True)
            path=package/'runtime.py';path.write_text('fixed = True\n')
            recorded={'code_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()}}
            check_evaluator_package(recorded,source)
            with self.assertRaisesRegex(ValueError,'differs from startup'):
                check_evaluator_package(recorded,source/'other')
            path.write_text('fixed = False\n')
            with self.assertRaisesRegex(ValueError,'source changed'):check_evaluator_package(recorded,source)


if __name__=='__main__':unittest.main()
