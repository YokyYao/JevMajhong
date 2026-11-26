import copy
from types import CodeType,SimpleNamespace
import unittest
from tools import curriculum_placement_canonical_parent_v2 as course
from tools import seal_curriculum_placement_retained_parent_v1 as original
from tools import curriculum_placement_retained_parent_v1 as retained
from tests.core.test_curriculum_placement_retained_seal_v1 import phases,data_fixture


def phase_fixture():
    cap,queue,captured,receipt,receipts=phases()
    for proof in cap['scheduler_proofs'].values():
        proof['raw']=proof['raw'].replace(str(retained.ENTRY.resolve()),str(course.ENTRY.resolve()))
    captured['raw']=captured['raw'].replace(str(retained.ENTRY.resolve()),str(course.ENTRY.resolve()))
    receipts={path.with_name(path.name.replace('curriculum-yaku-parent-exit-',course.PREFIX)):value
        for path,value in receipts.items()}
    return cap,queue,captured,receipt,receipts


def shape(code,reverse=False):
    values=[]
    for value in code.co_consts:
        if isinstance(value,CodeType):value=shape(value,reverse)
        elif reverse and isinstance(value,str):value=next((old for old,new in course.PATHS.items() if value==new),value)
        values.append(value)
    return code.co_code,code.co_names,code.co_varnames,code.co_freevars,values


class CanonicalPlacementParentTests(unittest.TestCase):
    def test_exact_successful_canonical_prepare_fit_audit_capture_required(self):
        cap,queue,captured,receipt,receipts=phase_fixture()
        exits,paths=course.phase_function()(cap,queue,captured,receipt,receipts.__getitem__)
        self.assertEqual(set(exits),{'prepare','fit','audit','capture'})
        self.assertTrue(all(path.name.startswith(course.PREFIX) for path in paths))

    def test_old_producer_failed_capture_or_swapped_identity_rejected(self):
        cap,queue,captured,receipt,receipts=phases()
        with self.assertRaises(ValueError):course.phase_function()(cap,queue,captured,receipt,receipts.__getitem__)
        cap,queue,captured,receipt,receipts=phase_fixture()
        captured['raw']=captured['raw'].replace('JobState=COMPLETED','JobState=FAILED')
        with self.assertRaises(ValueError):course.phase_function()(cap,queue,captured,receipt,receipts.__getitem__)
        cap,queue,captured,receipt,receipts=phase_fixture();queue['fit_job']=999
        with self.assertRaises(ValueError):course.phase_function()(cap,queue,captured,receipt,receipts.__getitem__)

    def test_source_functions_change_only_declared_path_constants_and_globals(self):
        for function in (original.phase_proofs,original.parent_artifacts,original.seal,retained.verify_parent):
            self.assertEqual(shape(course.scoped(function).__code__,True),shape(function.__code__))
        self.assertIs(course.verifier_function().__globals__['require_parent_evidence'],course.require_parent_evidence)

    def test_an_unreviewed_source_function_cannot_be_specialized(self):
        with self.assertRaises(ValueError):course.scoped(lambda:None)

    def test_actual_lineage_is_explicit(self):
        parent=dict(curriculum=dict(run_id='jev-curriculum-yaku-canonical-parent-v2',
            stage='stage3B-yaku-canonical-parent-v2',stage_offset=3468,
            out='runs/curriculum-yaku-canonical-parent-v2/stage3B-fit'),
            training_config=dict(parent_initialization=dict(method=course.canonical.METHOD,selected_step=2472)))
        course.require_lineage(parent)
        for key,value in (('run_id','old-balanced-parent'),('stage_offset',3396)):
            bad=copy.deepcopy(parent);bad['curriculum'][key]=value
            with self.assertRaises(ValueError):course.require_lineage(bad)
        bad=copy.deepcopy(parent);bad['training_config']['parent_initialization']['selected_step']=2400
        with self.assertRaises(ValueError):course.require_lineage(bad)

    def test_no_implicit_or_old_parent_seal(self):
        sealed=dict(placement_parent_lineage_method=course.METHOD,canonical_parent_selected_step=2472,
            placement_parent_selected_step=2500,parent_registration=str(course.PARENT))
        course.require_sealed_lineage(sealed)
        for key in sealed:
            bad=dict(sealed);del bad[key]
            with self.assertRaises((KeyError,ValueError)):course.require_sealed_lineage(bad)
        with self.assertRaises(ValueError):course.require_sealed_lineage(dict(sealed,parent_registration=str(original.PARENT)))

    def test_full57_data_requirement_is_unchanged(self):
        original.require_data(data_fixture(),dict(status='placement_all_sources_including_targeted_win_special_prepared_not_fit'))
        bad=data_fixture();bad['source_unique_records'].pop(next(iter(bad['source_unique_records'])))
        with self.assertRaises(ValueError):original.require_data(bad,dict(status='placement_all_sources_including_targeted_win_special_prepared_not_fit'))

    def test_parent_loader_restores_global_after_failure(self):
        previous=object();initialization=SimpleNamespace(verified_parent=previous)
        with self.assertRaises(RuntimeError):
            with course.parent_loader(initialization):
                self.assertIs(initialization.verified_parent,course.verify_parent)
                raise RuntimeError('stop')
        self.assertIs(initialization.verified_parent,previous)


if __name__=='__main__':unittest.main()
