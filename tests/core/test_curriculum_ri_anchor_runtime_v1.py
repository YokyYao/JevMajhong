import copy
import dis
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from tools import curriculum_ri_anchor_runtime_v1 as runtime
from tools import curriculum_ri_anchor_fit_v1 as lifecycle


def completed(passed=False):
    grade={'passed':passed,'development_only_not_independent_strength':True}
    done=dict(status='balanced_replay2472_fit_and_real_curves_complete',optimizer_steps=2472,actual_records_seen=180456)
    evidence=dict(status='balanced_replay2472_final_evidence_audit_passed',optimizer_steps=2472,
        saved_final_reloaded_menus=23688,stage3_advancement=grade)
    execution=dict(status='actual_balanced_replay2472_full_evidence_independently_audited',actual_fit_job='11')
    captured=dict(status='balanced_replay_actual_terminals_preserved_not_promoted',strategy_grade=copy.deepcopy(grade),
        scheduler_proofs={role:dict(state='COMPLETED',successful=True,exit_code='0:0',job_id=job)
            for role,job in (('fit','11'),('audit','12'))})
    return done,evidence,execution,captured


def recipe():
    old_args=dict(data='old_train',eval_data='old_val',out='old_out',max_steps=2472,
        checkpoint_every_steps=103,eval_every_steps=618,lr=3e-6,head_lr=1.5e-5,epochs=2,seed=2026112304)
    old=dict(curriculum=dict(base='base',base_revision='revision',epochs=2,accumulation=73,seed=2026112304,
        development_targets={'original_limit':.03}),parent_model={'directory':'qualified388'},
        source='frozen_project',source_sha256='source_hash',training_config=dict(base='base',base_revision='revision',
            parent_initialization={'method':'original','child_data_ready':'old_data'}))
    c=runtime.curriculum_view(old['curriculum']);args=runtime.trainer_view(old_args)
    cfg=runtime.config_view(old['training_config'],c,'trainer_hash','receipt_hash')
    reg=dict(registered_final_step=2596,train_records=92570,validation_records=24998,
        physical_train_slots_per_epoch=94754,expected_record_exposures=189508,course_loss_mass=runtime.MASS,
        validation_loss_method=runtime.VALIDATION_METHOD,curriculum=c,parent_model=old['parent_model'],
        source=old['source'],source_sha256=old['source_sha256'],training_config=cfg,
        integration_control_updates_excluded=True,
        replay_epoch_plans=[str(runtime.DATA/f'epoch-{n:02d}-private.json') for n in (1,2)])
    return reg,old,args,old_args


class AnchorRuntimeTests(unittest.TestCase):
    def test_only_successful_full_failed_strategy_can_trigger_alternative(self):
        values=completed();grade=runtime.failed_current(*values)
        self.assertFalse(grade['passed']);self.assertIsNot(grade,values[1]['stage3_advancement'])
        mutations=[(0,'optimizer_steps',1200),(0,'actual_records_seen',1000),
                   (1,'saved_final_reloaded_menus',100),(2,'actual_fit_job','999')]
        for index,key,value in mutations:
            bad=copy.deepcopy(values);bad[index][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):runtime.failed_current(*bad)
        bad=copy.deepcopy(values);bad[3]['scheduler_proofs']['audit']['exit_code']='1:0'
        with self.assertRaises(ValueError):runtime.failed_current(*bad)
        bad=copy.deepcopy(values);bad[3]['strategy_grade']['passed']=True
        with self.assertRaises(ValueError):runtime.failed_current(*bad)

    def test_qualified_current_preserves_its_original_later_course(self):
        values=completed(True)
        self.assertTrue(runtime.completed_current(*values)['passed'])
        with self.assertRaises(ValueError):runtime.failed_current(*values)

    def test_recipe_keeps_initializer_optimizer_and_original_quality_cohort(self):
        values=recipe();runtime.check_recipe(*values)
        for target,key,value in ((0,'train_records',92569),(0,'integration_control_updates_excluded',False),
                                  (2,'lr',1e-3)):
            bad=copy.deepcopy(values);bad[target][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):runtime.check_recipe(*bad)
        bad=copy.deepcopy(values);bad[0]['curriculum']['retention_quality_cohort']='combined'
        with self.assertRaises(ValueError):runtime.check_recipe(*bad)
        self.assertEqual(values[2]['eval_every_steps'],649)
        self.assertEqual(values[2]['checkpoint_every_steps'],59)

    def test_source_complete_schema_extension_keeps_original_numerical_code(self):
        extended=runtime.augmented_profile_function()
        self.assertEqual(extended.__code__,runtime.canonical_profile.profile.__code__)
        inner=extended.__globals__['previous_profile']
        original=runtime.original_profile.profile
        self.assertEqual(inner.__code__.co_code,original.__code__.co_code)
        self.assertEqual(inner.__code__.co_names,original.__code__.co_names)
        self.assertEqual(sum(i.opname=='LOAD_CONST' and i.argval==13789 for i in dis.get_instructions(inner)),1)
        self.assertEqual(sum(i.opname=='LOAD_CONST' and i.argval==12479 for i in dis.get_instructions(original)),1)

    def test_rollback_only_cancels_verified_jobs_from_own_script(self):
        observed={
            '11':SimpleNamespace(returncode=0,stdout=f'JobId=11 Command={lifecycle.ENTRY}'),
            '12':SimpleNamespace(returncode=0,stdout='JobId=12 Command=/another/project/fit.slurm'),
            '13':SimpleNamespace(returncode=1,stdout='')}
        with patch.object(lifecycle,'scheduler',side_effect=lambda job:observed[job]),patch.object(lifecycle.subprocess,'run') as cancel:
            lifecycle.cancel_owned(['11','12','13'])
        self.assertEqual(cancel.call_count,1)
        self.assertEqual(cancel.call_args.args[0],['scancel','11'])


if __name__=='__main__':unittest.main()
