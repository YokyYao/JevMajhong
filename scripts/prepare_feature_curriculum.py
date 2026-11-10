"""Mix fresh expert reactions and retained ordinary states, using public call features."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from call_policy import digest, request_hash, is_call_request
from call_option_features import enrich_request, VERSION
from clean_policy_data import policy_request
from head_rl import write
from checkpoint_identity import completed_checkpoint_files


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());data=ROOT/cfg['data'];out=ROOT/cfg['out']
    if data.exists() or out.exists():raise ValueError('Use new immutable data/run paths')
    if cfg['feature_version']!=VERSION:raise ValueError('Feature version differs')
    ordinary_cfg=json.loads((ROOT/cfg['source_ordinary_config']).read_text())
    ordinary=ROOT/ordinary_cfg['data'];old=json.loads((ordinary/'provenance.json').read_text())
    expert_cfg=json.loads((ROOT/cfg['source_expert_config']).read_text());expert=ROOT/expert_cfg['data']
    fresh=json.loads((expert/'provenance.json').read_text())
    if not old['complete'] or not fresh['complete'] or not fresh['chronological_dora_and_identified_passes']:
        raise ValueError('Source preparation incomplete')
    records={};source_hashes={}
    for split in ('train','validation'):
        for source,provenance,name in ((ordinary,old,'ordinary'),(expert,fresh,'expert')):
            path=source/(split+'.jsonl')
            if digest(path)!=provenance['files'][split]:raise ValueError('Source data changed')
            rows=[json.loads(l) for l in path.read_text().splitlines()]
            if name=='ordinary':rows=[r for r in rows if r['_meta']['phase']!='reaction']
            elif not all(is_call_request(r) for r in rows):raise ValueError('Expert reaction scope differs')
            records[name,split]=rows;source_hashes[name+'/'+split]=digest(path)
    if [len(records[k]) for k in [('ordinary','train'),('expert','train'),('ordinary','validation'),('expert','validation')]]!=[8192,32768,384,1024]:
        raise ValueError('Unexpected mixture coverage')
    combined={s:records['ordinary',s]+records['expert',s] for s in ('train','validation')}
    groups={s:{r['_meta']['group_id'] for r in rows} for s,rows in combined.items()}
    if groups['train']&groups['validation']:raise ValueError('Whole-match train/validation overlap')
    sealed=set(fresh['matches']['test'])
    if (groups['train']|groups['validation'])&sealed:raise ValueError('Sealed test match overlap')
    parent_cfg=json.loads((ROOT/cfg['clean_parent_config']).read_text());parent=ROOT/parent_cfg['out']
    protocol=json.loads((parent/'protocol.json').read_text())
    for name,h in protocol['code_sha256'].items():
        if digest(ROOT/name)!=h:raise ValueError('Parent source changed')
    initial=json.loads((parent/'initialization-audit.json').read_text())
    if not initial['fresh_lora_b_zero'] or initial['warm_start_requested']:raise ValueError('Parent is not clean origin')
    warm=ROOT/cfg['warm_start']
    if warm.resolve()!=(parent/'training-run/checkpoints/step-000200').resolve():raise ValueError('Unregistered initialization')
    warm_files=completed_checkpoint_files(warm,200)
    data.mkdir(parents=True);out.mkdir(parents=True)
    feature_counts={};counts={};phase_counts={}
    for split,rows in combined.items():
        labels=Counter();phases=Counter();n_features=0
        with (data/(split+'.jsonl')).open('w') as f:
            for record in rows:
                if is_call_request(record) and record['_meta'].get('view') not in (None,'native'):
                    raise ValueError('Only native reactions admitted')
                q=record['questions']['action'];before_keys=list(q['criteria']);gold=q['label']
                result=enrich_request(policy_request(record))
                if result['state']!=record['state'] or list(result['questions']['action']['criteria'])!=before_keys or result['questions']['action']['label']!=gold:
                    raise ValueError('Feature transform changed state/options/label')
                kind=q['criteria'][gold]['type'];phase=record['_meta'].get('phase','reaction')
                result['_meta'].update(id=record['_meta']['id']+':call_features',parent_id=record['_meta']['id'],phase=phase,view='native',feature_version=VERSION)
                result['_meta']['text_sha256']=request_hash(result)
                labels[kind]+=1;phases[phase]+=1;n_features+=is_call_request(result)
                f.write(json.dumps(result,ensure_ascii=False)+'\n')
        counts[split]=dict(labels);phase_counts[split]=dict(phases);feature_counts[split]=n_features
    report={'complete':True,'config':cfg,'records':{s:len(rows) for s,rows in combined.items()},
        'matches':{s:sorted(v) for s,v in groups.items()},'action_types':counts,'phase_counts':phase_counts,
        'feature_records':feature_counts,'source_files':source_hashes,'source_expert_provenance_sha256':digest(expert/'provenance.json'),
        'files':{s:digest(data/(s+'.jsonl')) for s in combined},'model_test_opened':False,
        'parent_protocol_sha256':digest(parent/'protocol.json'),'parent_result_sha256':digest(parent/'result.json'),
        'parent_initialization_sha256':digest(parent/'initialization-audit.json'),'warm_start_files':warm_files,
        'reference_files':protocol['reference_files'],'semantic_sha256':digest(ROOT/'semantic.py'),
        'repair_sha256':digest(ROOT/'replay_call_repair.py'),'dora_repair_sha256':digest(ROOT/'replay_dora_repair.py'),
        'policy_data_sha256':digest(ROOT/'clean_policy_data.py'),'feature_code_sha256':digest(ROOT/'call_option_features.py'),
        'guide_code_sha256':digest(ROOT/'guide_call_features.py'),'feature_version':VERSION,
        'scope':'native only; labels preserved; fresh natural-prior expert reactions plus ordinary retention; no synthetic strategy targets'}
    write(data/'provenance.json',report);write(out/'preparation.json',report)
    print(json.dumps({k:report[k] for k in ('complete','records','action_types','phase_counts','feature_records')}),flush=True)


if __name__=='__main__':main()
