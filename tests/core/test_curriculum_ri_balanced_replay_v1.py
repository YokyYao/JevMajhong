import copy
import ast
from pathlib import Path
import unittest
from tools import curriculum_ri_balanced_replay_runtime_v1 as runtime
from tests.core.test_curriculum_ri_replay_epoch_adapter_v1 import EpochAdapterTests


class BalancedFitTests(unittest.TestCase):
    def control(self, passed=False):
        done = dict(status='progression2400_fit_and_real_curves_complete', optimizer_steps=2400,
            actual_records_seen=175190)
        grade = dict(passed=passed, development_only_not_independent_strength=True)
        evidence = dict(status='progression2400_final_evidence_audit_passed', optimizer_steps=2400,
            saved_final_reloaded_menus=23688, stage3_advancement=grade)
        execution = dict(status='actual_progression2400_full_evidence_independently_audited', actual_fit_job=123)
        captured = dict(status='parent_retention_actual_terminals_preserved_not_child_admitted',
            strategy_grade_overridden=False, original_strategy_grade=grade,
            scheduler_proofs={role:dict(successful=True, state='COMPLETED', exit_code='0:0', job_id=str(job))
                for role, job in [('fit',123),('audit',124)]})
        return [done, evidence, execution, captured]

    def test_only_actual_final_strategy_failure_opens_standby(self):
        self.assertFalse(runtime.failed_control(*self.control())['passed'])
        with self.assertRaises(ValueError): runtime.failed_control(*self.control(True))
        for index, key, value in [(0,'optimizer_steps',1800), (0,'actual_records_seen',0),
                (1,'saved_final_reloaded_menus',10), (2,'actual_fit_job',999)]:
            args=self.control(); args[index][key]=value
            with self.assertRaises(ValueError): runtime.failed_control(*args)

    def test_infrastructure_failure_is_not_a_strategy_failure(self):
        for role in ('fit','audit'):
            for key, value in [('successful',False),('state','TIMEOUT'),('exit_code','124:0')]:
                args=self.control(); args[3]['scheduler_proofs'][role][key]=value
                with self.assertRaises(ValueError): runtime.failed_control(*args)
        args=self.control(); args[3]['original_strategy_grade']={'passed':True}
        with self.assertRaises(ValueError): runtime.failed_control(*args)

    def test_unique_admission_survives_base_trainer_guard(self):
        rows, plans, replay, args = EpochAdapterTests().fixture()
        n=len(rows); original=dict(requested_records=n,admitted_records=n,rejected_records=0,rejected=[])
        pool, unique, expanded=runtime.expand_after_unique_admission(replay,rows,original)
        # This is the actual pinned base_lora admission guard. Physical slots
        # differ from original rows, so returning the expansion report fails it.
        self.assertFalse(unique['rejected_records'])
        self.assertEqual(unique['admitted_records'],n)
        self.assertEqual(unique['requested_records'],n)
        self.assertNotEqual(len(pool),n)
        self.assertEqual(expanded['admitted_records'],len(pool))
        self.assertEqual(unique['physical_exposure_slots'],len(pool))
        with self.assertRaises(ValueError): runtime.expand_after_unique_admission(replay,rows,dict(original,rejected_records=1))

    def test_only_explicit_schedule_changes_and_gates_unchanged(self):
        old = dict(curriculum={'development_targets':{'floor':.6},'seed':2026112304},
            training_config={'base_revision':'fixed','lr':3e-6,'parent_initialization':{'step':388}},
            source='pinned',source_sha256='abc',parent_model={'identity':'parent'},course_loss_weights={'fixed':1})
        old_args={'init_from':'parent388','lr':3e-6,'head_lr':1.5e-5,'epochs':2,'batch':1,'accum':73}
        c=runtime.curriculum_view(old['curriculum'])
        cfg=dict(old['training_config'],model_id=c['run_id'],out=c['out'],
            trainer_config=str(runtime.RUN/'fit-plan/trainer.toml'),trainer_config_sha256='sha',
            registered_final_step=2472,loss_method=runtime.METHOD)
        reg=dict(status='balanced_replay2472_registered_not_started',registered_final_step=2472,
            train_records=87595,validation_records=23688,physical_train_slots_per_epoch=90228,
            expected_record_exposures=180456,course_loss_mass=runtime.MASS,curriculum=c,training_config=cfg,
            validation_loss_method=runtime.VALIDATION_METHOD,standby_only_after_control_failed_strategy=True,
            **{k:old[k] for k in ('source','source_sha256','parent_model','course_loss_weights')})
        args=runtime.trainer_view(old_args); runtime.check_matched(reg,old,args,old_args)
        bad=copy.deepcopy(reg); bad['curriculum']['development_targets']['floor']=.4
        with self.assertRaises(ValueError): runtime.check_matched(bad,old,args,old_args)
        with self.assertRaises(ValueError): runtime.check_matched(reg,old,dict(args,init_from='interim1200'),old_args)
        with self.assertRaises(ValueError): runtime.check_matched(reg,old,dict(args,resume='state'),old_args)
        self.assertTrue(all(p%103==0 for p in runtime.POINTS))

    def test_actual_final_grade_computation_is_unchanged(self):
        root=Path(__file__).resolve().parents[2]
        def grade_ast(name):
            module=ast.parse((root/'tools'/name).read_text())
            body=next(n.body for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='audit')
            start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='limits' for t in n.targets))
            end=next(i for i in range(start,len(body)) if isinstance(body[i],ast.For))
            return [ast.dump(n,include_attributes=False) for n in body[start:end]]
        self.assertEqual(grade_ast('audit_curriculum_progression_parent_v1.py'),
            grade_ast('audit_curriculum_ri_balanced_replay_v1.py'))


if __name__=='__main__': unittest.main()
