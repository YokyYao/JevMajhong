"""Frozen recipe, retention checks and v2 parent bridge for repaired calls."""
from contextlib import contextmanager
import math
from pathlib import Path

from tools.build_curriculum_calls_repair import MASS,layout
from tools.curriculum_basic_repair import grade
from tools.curriculum_transfer import METHOD,ROOT,read,sha,verify_curriculum_parent
from tools.curriculum_parent import ParentNotReady,rows
from tools.seal_curriculum_basic_repair import require_source_quality


def require_recipe(c,data,counts):
    expected=layout(counts)
    if (c.get('stage')!='stage2-calls-repair-v1' or c.get('batch')!=1 or c.get('epochs')!=2
        or c.get('optimizer_resume') is not False or c.get('new_optimizer') is not True
        or c.get('same_adapter_and_pointer_head') is not True
        or c.get('all_twelve_strata_each_update') is not True
        or c.get('each_epoch_without_replacement') is not True
        or data.get('status')!='calls_repair12_source_data_prepared_not_parent_or_fit_admitted'
        or data.get('source_specific_basic_parent_seal_required') is not True
        or data.get('required_input_schema') is not None
        or data.get('loss_mass')!=MASS or c.get('loss_mass')!=MASS
        or data.get('categories',{}).get('train')!=counts
        or data.get('count_only_accumulation')!=expected['accumulation']
        or data.get('example_weights')!=expected['example_weights']
        or c.get('accumulation')!=expected['accumulation']
        or c.get('minimum_tail_fraction')!=.75
        or c.get('validation_every_steps')!=expected['updates_per_epoch']
        or c.get('game_evaluation_every_steps')!=expected['updates_per_epoch']
        or c.get('candidate_rule')!='registered_final_full_two_epochs_calls_repair_v1'
        or c.get('learning_rate')!=1e-5 or c.get('head_learning_rate')!=5e-5
        or c.get('automatic_stage3_launch') is not False or c.get('automatic_release') is not False):
        raise ValueError('Prospective complete-epoch repaired-calls recipe changed')
    steps=2*expected['updates_per_epoch']
    if sum(counts.values())!=48710 or steps!=384:
        raise ValueError('This version freezes the actual audited48710-row data exposure')
    return dict(expected,final_step=steps,record_exposures=2*sum(counts.values()))


def balanced(counts):
    yes=counts.get('tp',0)+counts.get('fn',0);no=counts.get('tn',0)+counts.get('fp',0)
    if not yes or not no:raise ValueError('Both actual operation classes required; no invented recall')
    return .5*(counts.get('tp',0)/yes+counts.get('tn',0)/no)


def retention(first,last,limits):
    a=first['basic_replay'];b=last['basic_replay'];quality=grade(b)
    plain={source:b['source_stratum_accuracy'][source+':plain']-a['source_stratum_accuracy'][source+':plain'] for source in ('legacy','teacher')}
    riichi={source:balanced(b['riichi_by_source'][source])-balanced(a['riichi_by_source'][source]) for source in ('legacy','teacher')}
    passed=(quality['all_source_quality_passed']
        and min(plain.values())>=-limits['max_plain_agreement_drop_from_step0']
        and min(riichi.values())>=-limits['max_riichi_balanced_accuracy_drop_from_step0'])
    return dict(passed=passed,basic_source_quality=quality,plain_gain_by_source=plain,riichi_balanced_gain_by_source=riichi,
        development_only_not_policy_strength=True)


def verify_parent(cfg):
    spec=cfg['parent_initialization'];path=Path(spec['manifest']).resolve()
    if sha(path)!=spec['sha256']:raise ParentNotReady('Parent seal changed')
    seal=read(path)
    if (spec['method']!=METHOD or seal.get('method')!=METHOD
        or seal.get('source_specific_parent_quality_checked') is not True
        or seal.get('source_specific_sealer')!='tools/seal_curriculum_basic_repair_v2.py'):
        raise ParentNotReady('Completed v2 twelve-source qualified parent required; no research override')
    parent=verify_curriculum_parent(cfg)
    reg=read(seal['parent_registration']);validation=rows(ROOT/reg['curriculum']['out']/'metrics/validation_metrics.jsonl')[-1]
    quality=require_source_quality(reg,read(seal['parent_evidence']),validation)
    if quality!=seal.get('basic_repair_source_quality'):raise ParentNotReady('Actual final source grade differs')
    for name in ('tools/seal_curriculum_basic_repair_v2.py','tools/seal_curriculum_basic_repair.py','tools/curriculum_basic_repair.py'):
        p=(ROOT/name).resolve()
        if seal['bound_files'].get(str(p))!=sha(p):raise ParentNotReady('Source sealer/grade code is not bound')
    return parent


@contextmanager
def parent_loader(initialization):
    old=initialization.verified_parent;initialization.verified_parent=verify_parent
    try:yield
    finally:initialization.verified_parent=old
