"""TRAIN-only physical exposure mapping for the prospective weighted replay.

Resolve each virtual slot through its own epoch plan before invoking the pinned
trainer's augmentation/encoder. Attach loss weights after token encoding. This
adapter does not register, initialize, train, resume or select a model.
"""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_ri_inverse_repeat_loss_v1 import attach,FIELD,MARKER as WEIGHT_MARKER
from tools.curriculum_ri_every_update_replay_v1 import MASS,RI

METHOD='epoch_specific_original_replay_mapping_before_pinned_encode_batch_v1'
POSITION='_RI_replay_virtual_position_v1'


class EpochReplay:
    def __init__(self,rows,plans):
        self.rows=rows;self.plans=plans;self.started=0
        ids=[r['_meta']['id'] for r in rows];categories=[r['questions']['action']['src'] for r in rows]
        counts=Counter(categories)
        if (len(plans)!=2 or len(set(ids))!=len(rows) or set(counts)!=set(MASS)
                or any(POSITION in r['_meta'] for r in rows)):
            raise ValueError('Two complete plans over unique original19-source TRAIN rows required')
        self.batch=plans[0]['summary']['batch'];self.size=len(plans[0]['indices'])
        self.cursors=[0,0];self.consumed=[Counter(),Counter()];self.unique=[set(),set()]
        self.hashers=[hashlib.sha256(),hashlib.sha256()]
        for plan in plans:
            order=plan['indices'];repeats=Counter(order);weights=plan['weights_by_original_index']
            if (len(order)!=self.size or len(order)%self.batch or plan['summary']['batch']!=self.batch
                    or any(type(i) is not int or not 0<=i<len(rows) for i in order)
                    or set(repeats)!=set(range(len(rows))) or set(weights)!={str(i) for i in range(len(rows))}):
                raise ValueError('Complete fixed-size exposure plans required')
            for i in range(len(rows)):
                weight=weights[str(i)]
                if (type(weight) not in (int,float) or not math.isfinite(weight) or not 0<weight<=16
                        or abs(weight*repeats[i]/self.size-MASS[categories[i]]/counts[categories[i]])>1e-12):
                    raise ValueError('Actual per-original inverse-repeat mass required')
        self.virtual_requests=[]
        for position,i in enumerate(plans[0]['indices']):
            row=dict(rows[i]);row['_meta']=dict(rows[i]['_meta'],**{POSITION:position})
            self.virtual_requests.append(row)

    def admit(self,admitted,report):
        if (admitted!=self.rows or report.get('requested_records')!=len(self.rows)
                or report.get('admitted_records')!=len(self.rows) or report.get('rejected_records')!=0
                or report.get('rejected')!=[]):
            raise ValueError('Actual unchanged unique-row admission must precede expansion')
        return self.virtual_requests,dict(requested_records=self.size,admitted_records=self.size,
            rejected_records=0,rejected=[],unique_original_records=len(self.rows),
            unique_source_admission=copy.deepcopy(report),method=METHOD)

    def shuffle(self,order):
        if (self.started>=2 or sorted(order)!=list(range(self.size))
                or self.started and self.cursors[self.started-1]!=self.size):
            raise ValueError('Exactly two complete, ordered exposure epochs required')
        order[:]=range(self.size);self.started+=1

    def encode_batch(self,original,model,tok,args,chunk,epoch):
        if (type(epoch) is not int or epoch not in (0,1) or self.started!=epoch+1
                or args.batch!=1 or args.accum!=self.batch or getattr(args,'rlcr',False)
                or any(getattr(args,key)!=0 for key in ('p_none','p_none_distract','p_distract','p_none_pair'))):
            raise ValueError('Explicit supervised single-record fixed replay epoch required')
        positions=[r['_meta'].get(POSITION) for r in chunk]
        if (not positions or positions!=list(range(self.cursors[epoch],self.cursors[epoch]+len(chunk)))
                or any(type(p) is not int or not 0<=p<self.size for p in positions)):
            raise ValueError('No missing, repeated or stale virtual positions permitted')
        indices=[self.plans[epoch]['indices'][p] for p in positions]
        resolved=[self.rows[i] for i in indices]
        variants=original(model,tok,args,resolved,epoch)
        if len(variants)!=len(resolved):raise ValueError('Unregistered augmentation changed exposure count')
        for i,raw,variant in zip(indices,resolved,variants):
            questions=variant.rec['questions'];source=raw['questions']['action']['src']
            if (len(questions)!=1 or questions[0]['src']!=source
                    or questions[0]['keys'][questions[0]['label']]!=raw['questions']['action']['label']):
                raise ValueError('Epoch resolution changed source or observed action label')
            weight=self.plans[epoch]['weights_by_original_index'][str(i)]
            attach(variant.rec,weight,raw['_meta']['id'])
            self.consumed[epoch][source]+=1;self.unique[epoch].add(raw['_meta']['id'])
            self.hashers[epoch].update(json.dumps([i,weight],separators=(',',':')).encode()+b'\n')
        self.cursors[epoch]+=len(chunk)
        return variants

    def finish(self):
        if self.started!=2 or self.cursors!=[self.size,self.size]:raise ValueError('Both actual exposure epochs must finish')
        expected=[dict(Counter(self.rows[i]['questions']['action']['src'] for i in p['indices'])) for p in self.plans]
        if [dict(c) for c in self.consumed]!=expected or any(len(v)!=len(self.rows) for v in self.unique):
            raise ValueError('Actual source counts or original-row coverage differ')
        return dict(method=METHOD,epochs=2,physical_exposures=sum(self.cursors),
            unique_rows_each_epoch=[len(v) for v in self.unique],source_counts_by_epoch=expected,
            actual_mapping_weight_hashes=[h.hexdigest() for h in self.hashers])


def random_proxy(original,seed,adapter):
    created=[]
    class FixedReplayRandom:
        def __init__(self):self.base=original.Random(seed)
        def shuffle(self,order):adapter.shuffle(order)
        def __getattr__(self,name):return getattr(self.base,name)
    def factory(value=None):
        if type(value) is int and value==seed:
            if created:raise ValueError('Unexpected second global replay RNG')
            instance=FixedReplayRandom();created.append(instance);return instance
        return original.Random(value)
    return SimpleNamespace(seed=original.seed,Random=factory),created


def preflight():
    from tools.capture_curriculum_paired_terminal_v2 import resource
    from tools.curriculum_transfer import read,sha
    from tools.curriculum_calls_post_reach_data_v1 import write
    from tools import curriculum_progression_parent_loss_v1 as retained
    allocation=resource();parent=ROOT/'runs/curriculum-progression-parent-retention-v1'
    registration=parent/'fit-plan/registered.json';reg=read(registration)
    lossroot=ROOT/'runs/curriculum-ri-inverse-repeat-loss-v1'
    binding_path=lossroot/'completion-binding-private.json';binding=read(binding_path)
    if any(sha(p)!=h for p,h in binding['files'].items()):raise ValueError('Frozen loss/data prerequisite changed')
    out=ROOT/'runs/curriculum-ri-replay-epoch-adapter-v1'
    if out.exists():raise ValueError('Never replace an epoch adapter attempt')
    out.mkdir()
    sys.path[:0]=[reg['source'],reg['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    from jevany import train as trainer
    receipt=read(reg['training_config']['data_receipt']);path=Path(receipt['files']['train']['path'])
    with path.open() as stream:rows=[json.loads(line) for line in stream]
    planroot=ROOT/'runs/curriculum-ri-every-update-replay-v1'
    plans=[read(planroot/f'epoch-{epoch:02d}-private.json') for epoch in (1,2)]
    adapter=EpochReplay(rows,plans)
    admission_path=parent/'stage3-fit/token-admission.json'
    virtual,expanded_admission=adapter.admit(rows,read(admission_path))
    original=dict(materialize=trainer.materialize,random=trainer.random)
    proxy,created=random_proxy(original['random'],reg['curriculum']['seed'],adapter)
    args=SimpleNamespace(seed=reg['curriculum']['seed'],batch=1,accum=73,rlcr=False,
        p_none=0.,p_none_distract=0.,p_distract=0.,p_none_pair=0.,
        max_state=4096,max_branch=8192,max_packed=8192)
    class EncoderSpy:
        calls=0
        def encode(self,tok,record,**kwargs):
            if any(WEIGHT_MARKER in q or FIELD in q for q in record['questions']):
                raise ValueError('Weights were attached before the pinned encoder')
            self.calls+=1
            # Explicit stub IDs: this checks dispatch, not actual token admission.
            return {'ids':[1]}
    model=EncoderSpy();checked=0
    try:
        trainer.materialize=lambda r:retained.annotate(original['materialize'](r),r)
        trainer.random=proxy;rng=trainer.random.Random(args.seed);order=list(range(len(virtual)))
        for epoch in (0,1):
            rng.shuffle(order)
            for position in order:
                batch=adapter.encode_batch(trainer.encode_batch,model,None,args,[virtual[position]],epoch)
                q=batch[0].rec['questions'][0];i=plans[epoch]['indices'][position]
                if (q[FIELD]!=plans[epoch]['weights_by_original_index'][str(i)]
                        or q['_RI_inverse_repeat_original_id_v1']!=rows[i]['_meta']['id']):
                    raise ValueError('Actual pinned-batch dispatch attached the wrong row or weight')
                checked+=1
                if (position+1)%30000==0:print(json.dumps({'epoch':epoch+1,'pinned_batches_checked':position+1}),flush=True)
        result=adapter.finish()
        if len(created)!=1 or checked!=180456 or model.calls!=checked:raise ValueError('Actual complete pinned-batch mapping required')
    finally:
        trainer.materialize=original['materialize'];trainer.random=original['random']
    report=dict(status='actual_pinned_encode_batch_epoch_mapping_preflight_passed_not_model_training',
        **result,pinned_batches_checked=checked,encoder_spy_calls=model.calls,
        pinned_augmentation_and_materializer_used=True,model_encoder_was_stubbed=True,
        actual_new_full_token_admission_not_claimed=True,original_actual_unique_admission_reused=True,
        model_loaded=False,student_updates=0,new_games=0,allocation=allocation,
        live_trial_changed=False,fit_registration_and_actual_model_integration_pending=True)
    write(out/'summary-private.json',report);write(out/'expanded-admission-private.json',expanded_admission)
    files=dict(binding['files']);files[str(binding_path)]=sha(binding_path)
    for p in (Path(__file__).resolve(),ROOT/'tests/core/test_curriculum_ri_replay_epoch_adapter_v1.py',
        Path(trainer.__file__).resolve(),registration,admission_path,out/'summary-private.json',out/'expanded-admission-private.json'):
        files[str(p)]=sha(p)
    write(out/'completion-binding-private.json',{'files':files})
    print(json.dumps({k:v for k,v in report.items() if k not in ('allocation','source_counts_by_epoch','actual_mapping_weight_hashes')}),flush=True)


if __name__=='__main__':preflight()
