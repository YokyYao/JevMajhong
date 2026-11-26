"""Independent full calls audit with separated class objective/original action CE."""
import argparse
from collections import Counter,defaultdict
import gc
import json
import math
import os
import socket
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_fit import verify_parent,retention,balanced
from tools.curriculum_calls_riichi_fit_v1 import require_recipe,profile as source_profile,COMPONENTS,require_parent_kind
from tools.curriculum_calls_riichi_retention_v1 import METHOD
from tools.curriculum_calls_repair import require_parent_prediction_parity
from tools.curriculum_parent import require_replay


def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def audit(registered,out,binding=None):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
    assert not socket.gethostname().startswith(('lo-','login'))
    reg=read(registered);c=reg['curriculum'];fit=ROOT/c['out'];last=reg['registered_final_step']
    if binding:
        for p,h in read(binding).items():assert sha(p)==h,p
    for p,h in reg['inputs'].items():assert sha(p)==h,p
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(4)
    from safetensors.torch import load_file
    from jevmahjong.training.initialization import exact_loaded_tensors
    from jevmahjong.training.base_lora import target_cross_entropy
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevany.data import api_request
    from tools.curriculum_input_contract import verify_checkpoint,load_context_client
    from tools.curriculum_riichi_context import FIELD
    data=read(c['data_ready']);require_recipe(c,data,data['categories']['train'])
    assert reg['status']=='stage2_calls_riichi_retention_registered_not_started' and last==384
    assert reg['training_config']['loss_method']==METHOD
    parent=verify_parent(reg['training_config'])
    done=read(fit/'completion.json');assert done['status']=='stage2_calls_riichi_retention_fit_and_curves_complete' and done['loss_method']==METHOD
    assert done['optimizer_steps']==last and done['actual_records_seen']==reg['train_records']*c['epochs']
    init=read(fit/'initialization-audit.json');proof=read(fit/'actual-update-proof.json')
    assert init['parent_adapter_exact'] and init['parent_head_exact'] and init['new_optimizer']
    assert init['old_adapters_loaded'] and not init['pretrained_base_requires_grad'] and init['external_residual_networks']==0
    assert init['parent_identity_sha256']==c['parent_identity']
    assert proof['continued_LoRA_parameters_changed'] and proof['head_parameters_changed'] and not proof['cached_encoder_features_used']
    step0=fit/'training-run/checkpoints/step-000000'
    exact=exact_loaded_tensors(load_file(step0/'adapter_model.safetensors'),
        torch.load(step0/'head.pt',map_location='cpu',weights_only=True)['head'],parent['model']['directory'])
    assert exact['adapter_tensors']==372 and exact['head_tensors']==4
    warm=read(fit/'warmstart-validation-private.json')
    assert warm['exact_tensors']==exact and warm['checked_before_first_optimizer_update'] is True
    assert read(fit/'warmstart-games-private.json')==dict(matches=16,exact_parent_outcomes=True,checked_before_first_update=True)
    logs=rows(fit/'metrics/train_metrics.jsonl');assert [r['local_step'] for r in logs]==list(range(1,last+1))
    expected=[]
    for epoch in range(1,c['epochs']+1):
        receipt=read(fit/'epoch-order-private'/f'epoch-{epoch:02d}.json')
        assert receipt['each_record_exactly_once'] and receipt['copies_added']==0 and receipt['all_updates_have_all_declared_strata']
        expected.extend(receipt['source_counts_by_update'])
    assert expected==[r['source_counts'] for r in logs]
    counts=Counter()
    for row in logs:
        assert all(math.isfinite(row[k]) for k in ('loss','ce','grad_norm','lr')) and row['reward'] is None
        assert set(row['source_counts'])==set(reg['course_loss_mass'])
        assert row['loss_method']==METHOD and row['ce_is_original_hard_action_loss'] is True
        weighted=lambda name:sum(row['source_counts'][k]*row[name][k]*reg['course_loss_weights'][k] for k in row['source_counts'])/sum(row['source_counts'].values())
        assert abs(weighted('source_objective')-row['loss'])<1e-5
        assert abs(weighted('source_ce')-row['ce'])<1e-5
        for key,parts in row['source_loss_components'].items():
            assert set(parts)==set(COMPONENTS) and all(math.isfinite(value) for value in parts.values())
            assert parts['objective']==row['source_objective'][key] and parts['standard_action_ce']==row['source_ce'][key]
            expected=parts['riichi_max_class_ce']+parts['conditional_discard_ce']+parts['plain_action_ce']+parts['advanced_action_ce']
            assert abs(expected-parts['objective'])<1e-5
        counts.update(row['source_counts'])
    data=read(c['data_ready']);assert counts==Counter({k:v*c['epochs'] for k,v in data['categories']['train'].items()})
    receipt=read(reg['training_config']['data_receipt']);train=rows(ROOT/receipt['files']['train']['path']);validation=rows(ROOT/receipt['files']['validation']['path'])
    seal=read(c['parent_seal']);parent_reg=read(seal['parent_registration']);require_parent_kind(parent_reg)
    parent_files=read(parent_reg['training_config']['data_receipt'])['files']
    require_replay(rows(ROOT/parent_files['train']['path']),train);require_replay(rows(ROOT/parent_files['validation']['path']),validation)
    parent_fit=ROOT/parent_reg['curriculum']['out']
    parent_predictions=read(parent_fit/'training-run/training_eval'/f"step-{seal['selected_step']:06d}"/'development/rows.json')
    child_initial_predictions=read(fit/'training-run/training_eval/step-000000/development/rows.json')
    parity=require_parent_prediction_parity(rows(ROOT/parent_files['validation']['path']),parent_predictions,validation,child_initial_predictions)
    assert parity==warm['basic_validation_prediction_parity'] and parity['checked_basic_menus']==3266
    held={r['_meta']['group_id'] for r in validation}
    reserved=set(read(ROOT/parent_reg['curriculum']['data_root']/'reserved-public-groups.json'))
    assert not {r['_meta']['group_id'] for r in train}&(held|reserved)
    state=torch.load(fit/'training-run/checkpoints'/f'step-{last:06d}'/'trainer_state.pt',map_location='cpu',weights_only=False)
    assert state['optimizer_step']==last and state['epoch']==c['epochs'] and state['next_microbatch']==0
    assert state['records_seen']==len(train)*c['epochs'] and len(state['optimizer']['state'])==376
    assert state['world_size']==1 and sorted(state['order'].tolist())==list(range(len(train)))
    import hashlib
    final_order=read(fit/'epoch-order-private'/f"epoch-{c['epochs']:02d}.json")
    assert hashlib.sha256(json.dumps(state['order'].tolist()).encode()).hexdigest()==final_order['order_sha256']
    assert all(float(v['step'])==last for v in state['optimizer']['state'].values());del state;gc.collect()
    curve=read(fit/'validation-curve.json');steps=sorted({0,last,*range(c['validation_every_steps'],last+1,c['validation_every_steps'])})
    assert [r['step'] for r in curve]==steps
    byid={r['_meta']['id']:r for r in validation};assert len(byid)==len(validation)
    for point in curve:
        predictions=read(fit/'training-run/training_eval'/f"step-{point['step']:06d}"/'development/rows.json')
        assert len(predictions)==len(validation) and {r['id'] for r in predictions}==set(byid)
        recalculated=source_profile(validation,predictions,reg['course_loss_mass'],
            dict(weighted_target_ce=point['phase_weighted_ce']),target_cross_entropy)
        for k in ('stratum_ce','stratum_accuracy','stratum_counts','riichi_type_counts','riichi_by_source',
            'basic_replay','basic_replay_source_quality','operation_type_counts','operation_type_counts_by_source',
            'source_objective','source_loss_components','loss_method'):
            assert recalculated[k]==point[k],k
        assert abs(recalculated['weighted_target_ce']-point['weighted_target_ce'])<1e-12
        assert abs(recalculated['weighted_class_objective']-point['weighted_class_objective'])<1e-12
    journal=rows(fit/'metrics/validation_metrics.jsonl')
    assert [r['local_step'] for r in journal]==steps
    for event,point in zip(journal,curve):
        assert event['loss_method']==METHOD and event['loss']==point['weighted_class_objective'] and event['ce']==point['weighted_target_ce']
        assert event['source_objective']==point['source_objective'] and event['source_loss_components']==point['source_loss_components']
    assert curve[0]['basic_replay_source_quality']['all_source_quality_passed'] is True
    contracts={}
    if c.get('riichi_input_schema'):
        contracts={step:verify_checkpoint(fit/'training-run/checkpoints'/f'step-{step:06d}',reg,bundle) for step in steps}
        assert read(fit/'input-contracts.json')=={str(k):v for k,v in contracts.items()}
    checkpoint=fit/'training-run/checkpoints'/f'step-{last:06d}'
    client=UnifiedPolicyClient(load_model(checkpoint))
    if contracts:client=load_context_client(client,checkpoint,reg,bundle)
    saved={r['id']:r for r in read(fit/'training-run/training_eval'/f'step-{last:06d}'/'development/rows.json')};error=0.
    for row in validation:
        request=api_request(row)
        if contracts:
            for candidate in request['questions']['action']['criteria'].values():candidate.pop(FIELD,None)
        actual=client.system_one(**request)['answers']['action']['probabilities'];p=saved[row['_meta']['id']]
        assert set(actual)==set(p['keys']);error=max(error,max(abs(actual[k]-v) for k,v in zip(p['keys'],p['p'])))
    assert error<5e-5;del client;gc.collect();torch.cuda.empty_cache()
    eval_steps=sorted({0,last,*range(c['game_evaluation_every_steps'],last+1,c['game_evaluation_every_steps'])})
    assert done['curve_steps']==eval_steps
    games={};checked=0
    for label in ['reference-v8','reference-parent',*[f'step-{s:06d}' for s in eval_steps]]:
        folder=fit/'evaluations'/label;protocol=read(folder/'protocol.json');matches=rows(folder/'matches-private.jsonl')
        expected={(seed,profile,seat) for seed in protocol['seeds'] for profile in c['opponent_profiles'] for seat in range(4)}
        assert len(matches)==len(expected) and {(r['seed'],r['profile'],r['seat']) for r in matches}==expected
        expected_groups=c['stage_end_seed_groups'] if label in ('reference-v8','reference-parent',f'step-{last:06d}') else c['curve_seed_groups']
        assert protocol['seeds']==list(range(c['curve_seed_start'],c['curve_seed_start']+expected_groups))
        assert protocol['profiles']==c['opponent_profiles'] and protocol['seats']==4
        for row in matches:
            assert row['completed'] is True and row['ledger_passed'] is True
            ledger=settlement_ledger(row['mjai_log'],row['final_scores']);seat=row['seat']
            assert ledger['hands']==row['hands'] and row['net_points']==row['final_scores'][seat]-25000
            assert sorted(range(4),key=lambda p:(-row['final_scores'][p],p)).index(seat)+1==row['rank']
            assert hand_dealin_count(row['mjai_log'],seat)==row['dealin_hands']
            assert sum(e['type']=='hora' and e.get('actor')==seat for e in row['mjai_log'])==row['wins']
        summary=aggregate_complete_matches(matches);stored=read(folder/'summary.json')
        for key,value in summary.items():assert stored[key]==value
        identity=c['reference_identity'] if label=='reference-v8' else c['parent_identity'] if label=='reference-parent' else bundle(fit/'training-run/checkpoints'/label)['identity_sha256']
        assert stored['checkpoint_sha256']==identity
        assert protocol['checkpoint']['identity_sha256']==identity
        if len(matches)==64:assert stored['curve_grid_subset']==aggregate_complete_matches([r for r in matches if r['seed']<c['curve_seed_start']+c['curve_seed_groups']])
        if contracts and label.startswith('step-'):assert protocol['input_contract']==contracts[int(label.split('-')[1])]
        games[label]=matches;checked+=len(matches)
    parent_game_map={(r['seed'],r['profile'],r['seat']):r for r in games['reference-parent']}
    assert len(games['step-000000'])==16
    for r in games['step-000000']:
        reference=parent_game_map[r['seed'],r['profile'],r['seat']]
        assert all(r[k]==reference[k] for k in ('rank','net_points','final_scores','hands','wins','dealin_hands','choices'))
    final=games[f'step-{last:06d}'];comparisons={}
    for name in ('reference-v8','reference-parent'):
        reference={(r['seed'],r['profile'],r['seat']):r for r in games[name]};values={}
        for field,fn in [('top2',lambda r:float(r['rank']<=2)),('net_points',lambda r:r['net_points']),
                         ('hand_win',lambda r:r['wins']/r['hands']),('dealin',lambda r:r['dealin_hands']/r['hands']),('fourth',lambda r:float(r['rank']==4))]:
            clusters=[statistics.mean(fn(r)-fn(reference[r['seed'],r['profile'],r['seat']]) for r in final if r['seed']==seed)
                for seed in sorted({r['seed'] for r in final})]
            mean=statistics.mean(clusters);radius=2.3646242510102993*statistics.stdev(clusters)/math.sqrt(len(clusters))
            values[field]=dict(gain=mean,ci95=[mean-radius,mean+radius],seed_groups=len(clusters))
        comparisons[name]=values
    first,lastval=curve[0],curve[-1];limits=c['development_targets']
    basic_retention=retention(first,lastval,limits)
    plain_gain=lastval['basic_replay']['stratum_accuracy']['plain']-first['basic_replay']['stratum_accuracy']['plain']
    riichi_gain=balanced(lastval['riichi_type_counts'])-balanced(first['riichi_type_counts'])
    call_gain=statistics.mean(balanced(lastval['operation_type_counts'][k])-balanced(first['operation_type_counts'][k])
        for k in ('chi','pon','ankan','daiminkan','kakan'))
    advancement=dict(plain_agreement_gain=plain_gain,riichi_balanced_gain=riichi_gain,five_call_type_balanced_gain=call_gain,
        passed=basic_retention['passed']
            and call_gain>=limits['min_five_call_type_balanced_accuracy_gain_from_step0']
            and comparisons['reference-parent']['net_points']['gain']>=limits['min_final64_net_gain_over_parent']
            and comparisons['reference-parent']['dealin']['gain']<=limits['max_final64_dealin_increase_over_parent'],
        development_only_not_strength_proof=True)
    for ext in ('png','pdf'):assert (fit/'metrics'/f'curriculum-curves.{ext}').stat().st_size>1000
    plot=read(fit/'metrics/plot-data-provenance.json');assert plot['curve_steps']==eval_steps and plot['final64_steps']==[last]
    for p,h in reg['inputs'].items():assert sha(p)==h,p
    if binding:
        for p,h in read(binding).items():assert sha(p)==h,p
    result=dict(status='stage2_calls_riichi_retention_final_evidence_audit_passed',loss_method=METHOD,
        optimizer_steps=last,real_records_seen=len(train)*c['epochs'],
        actual_source_exposure=dict(counts),step0_exact_parent_tensors=exact,all376_optimizer_states_match=True,
        saved_final_reloaded_menus=len(validation),saved_final_max_probability_error=error,checked_complete_games=checked,
        final_summary=aggregate_complete_matches(final),comparison_to_v8=comparisons['reference-v8'],comparison_to_parent=comparisons['reference-parent'],
        original_parent_stage1_advancement=seal['original_stage1_advancement'],stage2_advancement=advancement,
        original_parent_basic_source_quality=seal['basic_repair_source_quality'],source_specific_basic_retention=basic_retention,
        actual_step0_basic_prediction_parity=parity,actual_step0_same16_parent_outcomes=True,
        heldout_groups_excluded=True,exact_train_and_validation_replay=True,source_and_data_hashes_unchanged=True,
        development_grid_not_independent_confirmation=True,independent_han_fu_scoring=False,
        automatic_stage3_started=False,automatic_release=False)
    write(out,result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--registered',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--binding',type=Path);a=p.parse_args()
    try:audit(a.registered,a.out,a.binding)
    except BaseException as error:
        write(a.out.with_name(a.out.stem+'-error.json'),dict(error_type=type(error).__name__,message=str(error)));raise
