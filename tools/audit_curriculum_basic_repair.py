"""Complete six-source fresh-base fit audit, including source-specific quality."""
from collections import Counter,defaultdict
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')

def audit(registered, destination, binding=None):
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS'),'Scheduled GPU required'
    import torch
    torch.set_num_threads(4)
    reg=read(registered);c=reg['curriculum'];out=ROOT/c['out'];last=reg['registered_final_step']
    if binding:
        for p,h in read(binding).items():assert sha(p)==h,p
    for p,h in reg['inputs'].items():assert sha(p)==h,p
    sys.path.insert(0,reg['source']);sys.path.insert(0,reg['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest,target_cross_entropy
    from jevmahjong.training.metric_journal import aggregate_complete_matches
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.metrics import hand_dealin_count
    from jevmahjong.policy.routing import load_model
    from jevmahjong.policy.unified import UnifiedPolicyClient
    from jevany.data import api_request
    assert package_digest(Path(reg['source'])/'jevmahjong')==reg['source_sha256']
    done=read(out/'completion.json');assert done['status']=='stage1_fit_and_curves_complete'
    assert done['optimizer_steps']==last and done['actual_records_seen']==c['epochs']*reg['train_records']
    init=read(out/'initialization-audit.json');assert init['fresh_lora_B_zero'] and not init['old_adapters_loaded']
    assert init['pretrained_base_requires_grad'] is False and init['external_residual_networks']==0
    proof=read(out/'actual-update-proof.json');assert proof['fresh_LoRA_B_now_nonzero'] and proof['head_parameters_changed']
    assert not proof['cached_encoder_features_used']
    updates=rows(out/'metrics/train_metrics.jsonl')
    assert [r['local_step'] for r in updates]==list(range(1,last+1))
    counts=Counter()
    for r in updates:
        assert all(math.isfinite(r[k]) for k in ('loss','ce','lr','grad_norm')) and r['reward'] is None
        counts.update(r['source_counts'])
        expected=sum(r['source_counts'][k]*r['source_ce'][k]*reg['stage1_loss_weights'][k] for k in r['source_counts'])/sum(r['source_counts'].values())
        assert abs(expected-r['loss'])<1e-5
    data=read(ROOT/c['data_root']/'data-ready.json')
    assert counts==Counter({k:c['epochs']*v for k,v in data['stage1_strata']['train'].items()})
    state=torch.load(out/'training-run/checkpoints'/f'step-{last:06d}'/'trainer_state.pt',map_location='cpu',weights_only=False)
    assert state['optimizer_step']==last and state['epoch']==c['epochs'] and state['next_microbatch']==0
    assert state['records_seen']==c['epochs']*reg['train_records'] and state['world_size']==1
    assert sorted(state['order'].tolist())==list(range(reg['train_records']))
    assert len(state['optimizer']['state'])==376
    assert all(float(v['step'])==last for v in state['optimizer']['state'].values())
    if c.get('balanced_epoch_order'):
        scheduled=[]
        for epoch in range(1,c['epochs']+1):
            proof=read(out/'epoch-order-private'/f'epoch-{epoch:02d}.json')
            assert proof['epoch']==epoch and proof['records']==reg['train_records']
            assert proof['each_record_exactly_once'] and proof['copies_added']==0
            assert proof['all_updates_have_all_declared_strata']
            scheduled.extend(proof['source_counts_by_update'])
        assert scheduled==[r['source_counts'] for r in updates]
        assert hashlib.sha256(json.dumps(state['order'].tolist()).encode()).hexdigest()==proof['order_sha256']
    del state;gc.collect()
    train=rows(ROOT/data['stages']['stage1']['train']['path']);validation=rows(ROOT/data['stages']['stage1']['validation']['path'])
    held={r['_meta']['group_id'] for r in validation}
    reserved=set(read(ROOT/c['data_root']/'reserved-public-groups.json'))
    for stage,parts in data['stages'].items():
        for split,spec in parts.items():assert sha(ROOT/spec['path'])==spec['sha256']
        groups={r['_meta']['group_id'] for r in rows(ROOT/parts['train']['path'])}
        assert not groups&(held|reserved)
    curve=read(out/'validation-curve.json');expected_steps=sorted({0,last,*range(c['validation_every_steps'],last+1,c['validation_every_steps'])})
    assert [r['step'] for r in curve]==expected_steps
    byid={r['_meta']['id']:r for r in validation}
    for point in curve:
        predictions=read(out/'training-run/training_eval'/f"step-{point['step']:06d}"/'development/rows.json')
        assert len(predictions)==len(validation) and {r['id'] for r in predictions}==set(byid)
        from tools.curriculum_basic_repair import profile,kind
        recalculated=profile(validation,predictions,reg['stage1_loss_mass'],dict(weighted_target_ce=point['natural_target_ce']),target_cross_entropy)
        for key in ('stratum_ce','stratum_accuracy','stratum_counts','source_stratum_ce','source_stratum_accuracy',
                    'source_stratum_counts','riichi_by_source','riichi_type_counts'):
            assert recalculated[key]==point[key],key
        assert abs(recalculated['weighted_target_ce']-point['weighted_target_ce'])<1e-12
    reloads=[]
    for step in (0,last):
        path=out/'training-run/checkpoints'/f'step-{step:06d}';client=UnifiedPolicyClient(load_model(path))
        saved={p['id']:p for p in read(out/'training-run/training_eval'/f'step-{step:06d}'/'development/rows.json')}
        sample=validation if step==last else [r for r in validation if kind(r['questions']['action']['src'].removeprefix('curriculum1:'))!='plain']+[
            r for r in validation if kind(r['questions']['action']['src'].removeprefix('curriculum1:'))=='plain'][:32]
        error=0.
        for r in sample:
            actual=client.system_one(**api_request(r))['answers']['action']['probabilities'];p=saved[r['_meta']['id']]
            assert set(actual)==set(p['keys'])
            error=max(error,max(abs(actual[k]-v) for k,v in zip(p['keys'],p['p'])))
        assert error<5e-5,error
        reloads.append(dict(step=step,menus=len(sample),max_probability_error=error,identity=bundle(path)['identity_sha256']))
        del client;gc.collect();torch.cuda.empty_cache()
    expected_eval=sorted({0,last,*range(c['game_evaluation_every_steps'],last+1,c['game_evaluation_every_steps'])})
    assert done['curve_steps']==expected_eval
    all_rows={};checked=0
    for name in ['reference-v8',*[f'step-{s:06d}' for s in expected_eval]]:
        folder=out/'evaluations'/name;protocol=read(folder/'protocol.json');games=rows(folder/'matches-private.jsonl')
        expected={(seed,profile,seat) for seed in protocol['seeds'] for profile in c['opponent_profiles'] for seat in range(4)}
        assert len(games)==len(expected) and {(r['seed'],r['profile'],r['seat']) for r in games}==expected
        for r in games:
            ledger=settlement_ledger(r['mjai_log'],r['final_scores']);seat=r['seat']
            assert ledger['hands']==r['hands'] and r['net_points']==r['final_scores'][seat]-25000
            assert sorted(range(4),key=lambda p:(-r['final_scores'][p],p)).index(seat)+1==r['rank']
            assert sum(e['type']=='hora' and e.get('actor')==seat for e in r['mjai_log'])==r['wins']
            assert hand_dealin_count(r['mjai_log'],seat)==r['dealin_hands']
        actual=aggregate_complete_matches(games);stored=read(folder/'summary.json')
        for k,v in actual.items():assert stored[k]==v,(name,k)
        if len(games)==64:
            subset=aggregate_complete_matches([r for r in games if r['seed']<c['curve_seed_start']+c['curve_seed_groups']])
            assert stored['curve_grid_subset']==subset
        if name=='reference-v8':assert stored['checkpoint_sha256']==c['reference_identity']
        else:
            step=int(name.split('-')[1]);assert stored['checkpoint_sha256']==bundle(out/'training-run/checkpoints'/name)['identity_sha256']
        all_rows[name]=games;checked+=len(games)
    reference={(r['seed'],r['profile'],r['seat']):r for r in all_rows['reference-v8']}
    final=all_rows[f'step-{last:06d}'];deltas={}
    for field,fun in [('top2',lambda r:float(r['rank']<=2)),('net_points',lambda r:r['net_points']),
                      ('hand_win',lambda r:r['wins']/r['hands']),('dealin',lambda r:r['dealin_hands']/r['hands']),
                      ('fourth',lambda r:float(r['rank']==4))]:
        clusters=[]
        for seed in sorted({r['seed'] for r in final}):
            clusters.append(statistics.mean(fun(r)-fun(reference[r['seed'],r['profile'],r['seat']]) for r in final if r['seed']==seed))
        mean=statistics.mean(clusters);radius=2.3646242510102993*statistics.stdev(clusters)/math.sqrt(len(clusters))
        deltas[field]=dict(gain=mean,ci95=[mean-radius,mean+radius],seed_groups=len(clusters))
    for ext in ('png','pdf'):assert (out/'metrics'/f'curriculum-curves.{ext}').stat().st_size>1000
    for kind in ('train','validation','evaluation'):assert (out/'metrics'/f'{kind}_metrics.csv').stat().st_size>0
    plot_proof=read(out/'metrics/plot-data-provenance.json')
    assert plot_proof['method']=='matched16_development_grid_v1' and plot_proof['curve_steps']==expected_eval
    assert plot_proof['final64_steps']==[last] and plot_proof['original_final64_results_unchanged']
    for p,h in reg['inputs'].items():assert sha(p)==h,p
    final_validation=curve[-1];rc=final_validation.get('riichi_type_counts')
    from tools.curriculum_basic_repair import grade
    source_quality=grade(final_validation);advancement=source_quality['original_aggregate_gate']
    if binding:
        for p,h in read(binding).items():assert sha(p)==h,p
    result=dict(status='stage1_final_evidence_audit_passed',optimizer_steps=last,real_records_seen=c['epochs']*reg['train_records'],
        actual_source_exposure=dict(counts),optimizer_states=376,heldout_groups_excluded_from_all_stages=True,
        saved_reloads=reloads,checked_complete_games=checked,final_comparison_to_v8=deltas,
        final_summary=aggregate_complete_matches(final),reference_summary=aggregate_complete_matches(all_rows['reference-v8']),
        development_grid_not_independent_confirmation=True,independent_han_fu_scoring=False,
        final64_subset_reconciled_and_curve_grid_aligned=True,
        source_and_data_hashes_unchanged=True,automatic_stage2_started=False,automatic_release=False,
        stage1_advancement=advancement,basic_repair_source_quality=source_quality,
        source_specific_parent_seal_required=True)
    write(destination,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--binding',type=Path)
    a=p.parse_args()
    try:audit(a.registered,a.out,a.binding)
    except BaseException as error:
        write(a.out.with_name(a.out.stem+'-error.json'),dict(error_type=type(error).__name__,message=str(error)))
        raise
