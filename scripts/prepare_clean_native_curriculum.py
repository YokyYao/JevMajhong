"""Native-first continuation using verified raw states from the clean data pool."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from call_policy import digest, request_hash
from clean_policy_data import policy_request
from head_rl import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());parent_cfg=json.loads((ROOT/cfg['clean_parent_config']).read_text())
    source=ROOT/parent_cfg['data'];parent=ROOT/parent_cfg['out'];out=ROOT/cfg['data'];run=ROOT/cfg['out']
    result=json.loads((parent/'result.json').read_text());provenance=json.loads((source/'provenance.json').read_text())
    if result['status']!='clean_policy_no_admissible_improvement' or result['optimizer_steps']!=200:
        raise ValueError('This continuation is registered for the completed clean200 trial only')
    if not provenance['complete'] or provenance['model_test_opened']:
        raise ValueError('Invalid source data')
    if out.exists() or run.exists():raise ValueError('Use a new data/run directory')
    parent_protocol=json.loads((parent/'protocol.json').read_text())
    for n,h in parent_protocol['code_sha256'].items():
        if digest(ROOT/n)!=h:raise ValueError('Parent code changed')
    for split in ('train','validation'):
        if digest(source/(split+'.jsonl'))!=provenance['files'][split]:raise ValueError('Parent data changed')
    final=[json.loads(l) for l in (source/'train.jsonl').read_text().splitlines()]
    raw=[json.loads(l) for l in (source/'train-raw.jsonl').read_text().splitlines()]
    by_parent={r['_meta']['parent_id']:r for r in final}
    if len(by_parent)!=len(raw) or len(raw)!=32768:raise ValueError('Raw/view identity coverage differs')
    for r in raw:
        other=by_parent[r['_meta']['id']]
        if request_hash(r)!=r['_meta']['text_sha256'] or request_hash(policy_request(r,other['_meta']['view']))!=other['_meta']['text_sha256']:
            raise ValueError('Raw record differs from audited source view')
        if r['questions']['action']['label']!=other['questions']['action']['label']:
            raise ValueError('Raw/source expert labels differ')
    selected=[]
    for phase,n in cfg['native_phase_records'].items():
        pool=[r for r in raw if r['_meta']['phase']==phase]
        pool.sort(key=lambda r:hashlib.sha256((str(cfg['seed'])+r['_meta']['id']).encode()).hexdigest())
        if len(pool)<n:raise ValueError('Phase coverage insufficient')
        selected.extend(pool[:n])
    validation=[json.loads(l) for l in (source/'validation.jsonl').read_text().splitlines()]
    validation=[r for r in validation if r['_meta']['view']=='native']
    if len(validation)!=576:raise ValueError('Native validation coverage differs')
    matches={s:sorted({r['_meta']['group_id'] for r in rows}) for s,rows in [('train',selected),('validation',validation)]}
    if set(matches['train'])&set(matches['validation']):raise ValueError('Whole-match overlap')
    if not set(matches['train'])<=set(provenance['matches']['train']) or not set(matches['validation'])<=set(provenance['matches']['validation']):
        raise ValueError('Foreign source split')
    warm=ROOT/cfg['warm_start']
    if warm.resolve()!=(parent/'training-run/checkpoints/step-000200').resolve():raise ValueError('Unregistered warm checkpoint')
    names=('adapter_config.json','adapter_model.safetensors','head.pt','tokenizer.json','tokenizer_config.json','chat_template.jinja','trainer_state.json')
    warm_files={n:digest(warm/n) for n in names}
    initial=json.loads((parent/'initialization-audit.json').read_text())
    if not all(initial[k] for k in ('fresh_lora_b_zero','fresh_lora_a_nonzero','pointer_differs_from_frozenB')) or initial['warm_start_requested']:
        raise ValueError('Parent did not originate from clean pretrained base')
    out.mkdir(parents=True);run.mkdir(parents=True)
    with (out/'train.jsonl').open('w') as f:
        for r in selected:
            row=policy_request(r);row['_meta'].update(view='native',parent_id=r['_meta']['id'],id=r['_meta']['id']+':native_curriculum')
            row['_meta']['text_sha256']=request_hash(row);f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (out/'validation.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in validation))
    report={'complete':True,'config':cfg,'records':{'train':len(selected),'validation':len(validation)},
        'raw_records':{'train':len(selected),'validation':len(validation)},'matches':matches,
        'phase_counts':dict(Counter(r['_meta']['phase'] for r in selected)),
        'action_types':dict(Counter(r['questions']['action']['criteria'][r['questions']['action']['label']]['type'] for r in selected)),
        'files':{s:digest(out/(s+'.jsonl')) for s in ('train','validation')},
        'parent_provenance_sha256':digest(source/'provenance.json'),'parent_protocol_sha256':digest(parent/'protocol.json'),
        'parent_result_sha256':digest(parent/'result.json'),'parent_initialization_sha256':digest(parent/'initialization-audit.json'),
        'warm_start_files':warm_files,'reference_files':parent_protocol['reference_files'],
        'semantic_sha256':digest(ROOT/'semantic.py'),'repair_sha256':digest(ROOT/'replay_call_repair.py'),
        'dora_repair_sha256':digest(ROOT/'replay_dora_repair.py'),'policy_data_sha256':digest(ROOT/'clean_policy_data.py'),
        'model_test_opened':False,'new_independent_matches':0,
        'selection':'all8192 reactions, fixed-ID-hash4096 closed +4096 meld discards; no label quota; native only',
        'scope':'native skill curriculum; screenshot validation remains a later stage, no Windows deployment claim'}
    write(out/'provenance.json',report);write(run/'preparation.json',report)
    print(json.dumps({k:report[k] for k in ('complete','records','phase_counts','action_types','new_independent_matches')}),flush=True)


if __name__=='__main__':main()
