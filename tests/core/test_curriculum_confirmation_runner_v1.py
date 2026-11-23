import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.curriculum_confirmation_runtime_v1 import require_header,cells,require_output_isolation,require_bound_files,STATUS,REFERENCE
from tools.curriculum_confirmation_contract_v1 import CRITERIA
from tools.run_curriculum_confirmation_v1 import run_cells


def header():
    return dict(status=STATUS,criteria=copy.deepcopy(CRITERIA),game_mode='4p-red-east',
        rule='GameRule.default_mjsoul',seed_namespace='spaced-v1',role_order=['reference','candidate'],
        reference_model=dict(identity_sha256=REFERENCE),optimizer_updates_during_confirmation=0,
        reserved_grid_already_opened=False,automatic_release=False,allow_adaptive_extra_games=False,
        raw_matches_private=True,requires_independent_postrun_audit=True)


class ConfirmationRunnerTests(unittest.TestCase):
    def test_statistical_or_training_plans_and_changed_grid_cannot_admit_execution(self):
        require_header(header())
        for status in ('statistical_plan_only_not_candidate_or_execution_admitted',
            'placement_data_parent_and_budget_registered_not_fit_admitted'):
            with self.assertRaises(ValueError):require_header(dict(status=status))
        for key,value in (('optimizer_updates_during_confirmation',1),('reserved_grid_already_opened',True),
            ('role_order',['candidate','reference']),('allow_adaptive_extra_games',True)):
            bad=header();bad[key]=value
            with self.assertRaises(ValueError):require_header(bad)
        bad=header();bad['criteria']['seed_groups']=64
        with self.assertRaises(ValueError):require_header(bad)

    def test_only_enumerate_exact_predeclared_cells_without_simulating_them(self):
        scheduled=cells(CRITERIA)
        self.assertEqual(len(scheduled),256);self.assertEqual(len(set(scheduled)),256)
        self.assertEqual(scheduled[0],(2026300000,'efficiency',0))
        self.assertEqual(scheduled[-1],(2026300031,'calling',3))

    def test_fictitious_cell_driver_emits_one_complete_result_and_only_count_progress(self):
        scheduled=[(910000000,'efficiency',0),(910000001,'calling',2)]
        agent=object();seen=[];emitted=[];progress=[]
        def play(actual,seed,seat,profile):
            self.assertIs(actual,agent);seen.append((seed,profile,seat))
            return dict(seed=seed,seat=seat,profile=profile,completed=True,ledger_passed=True)
        rows=run_cells(agent,scheduled,play,emitted.append,lambda done,total:progress.append((done,total)))
        self.assertEqual(seen,scheduled);self.assertEqual(rows,emitted)
        self.assertEqual(progress,[(1,2),(2,2)])
        def wrong(actual,seed,seat,profile):
            return dict(seed=seed+1,seat=seat,profile=profile,completed=True,ledger_passed=True)
        with self.assertRaises(ValueError):run_cells(agent,scheduled,wrong,emitted.append,lambda *a:None)

    def test_final_outputs_cannot_be_created_inside_source_or_model(self):
        with TemporaryDirectory() as directory:
            parent=Path(directory)/'training';source=Path(directory)/'runtime'
            require_output_isolation(Path(directory)/'confirmation',[parent,source])
            for bad in (parent,parent/'weights/confirmation',source/'outputs'):
                with self.assertRaises(ValueError):require_output_isolation(bad,[parent,source])

    def test_parent_sources_must_remain_in_final_binding_instead_of_only_a_passing_grade(self):
        with TemporaryDirectory() as directory:
            path=str((Path(directory)/'ancestor.jsonl').resolve())
            require_bound_files({path:'fixed'},{path:'fixed'})
            with self.assertRaises(ValueError):require_bound_files({},{path:'fixed'})
            with self.assertRaises(ValueError):require_bound_files({path:'different'},{path:'fixed'})
