"""Combine every prespecified shard, then audit one fixed-sample primary result."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagonal_replication import combine_rows,shard_configs
from head_rl import paired_primary_comparison,summarize,write
from scripts.audit_score_confirmation import audit,require,sha256


def finish(out):
    protocol=json.loads((out/'protocol.json').read_text());config=protocol['config']
    selected=json.loads((out/'confirmation-selection.json').read_text());expected=selected['candidate_head_sha256']
    require(protocol['shards_predeclared']==2 and config['confirmation_seeds']==512,'Unexpected replication sample')
    collected={'reference':[],'candidate':[]};integrity=[]
    for cfg in shard_configs(config):
        shard=ROOT/cfg['out'];actual=json.loads((shard/'protocol.json').read_text())
        require(actual['config']==cfg,'Prespecified shard changed')
        require(actual['sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),'Shard fingerprint changed')
        check=audit(shard,expected);integrity.append({'run':str(shard),'audit_status':check['audit_status']})
        for name in collected:
            collected[name].append([json.loads(line) for line in (shard/f'confirmation-{name}'/'games.jsonl').read_text().splitlines()])
    rows={name:combine_rows(parts,config) for name,parts in collected.items()}
    paired=paired_primary_comparison(rows['candidate'],rows['reference'],config)
    candidate,reference=summarize(rows['candidate']),summarize(rows['reference'])
    interval=paired['primary_delta_ci95']
    qualified=(candidate['mean_net_points']>=max(config['target_mean_net_points'],reference['mean_net_points']+config['target_net_points_gain'])
               and interval[0]>0)
    # Stage the joined records. If an observer retries after a partial local
    # finalization, accept only byte-identical existing records, never overwrite.
    with tempfile.TemporaryDirectory(dir=out,prefix='.assemble-') as temporary:
        for name,data in rows.items():
            staged=Path(temporary)/f'confirmation-{name}';staged.mkdir()
            (staged/'games.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in data))
            write(staged/'summary.json',summarize(data));destination=out/staged.name
            if destination.exists():
                require(sha256(destination/'games.jsonl')==sha256(staged/'games.jsonl'),'Existing joined rows differ')
            else:staged.rename(destination)
    result={'status':'complete','qualified':qualified,'candidate':candidate,'reference':reference,
            'primary_metric':'mean_net_points','primary_delta_ci95':interval,'paired':paired,
            'game_mode':config['game_mode'],'checkpoint':selected['checkpoint'],'default_policy_changed':False,
            'note':'One frozen candidate, all512 fresh independent walls across2 prespecified shards; no shard selection or optional stopping.'}
    try:existing=json.loads((out/'confirmation-result.json').read_text())
    except FileNotFoundError:existing=None
    if existing is not None:require(existing==result,'Existing complete replication result differs')
    else:write(out/'confirmation-result.json',result)
    report=audit(out,expected);report['shard_integrity']=integrity
    write(out/'confirmation-audit.json',report)
    write(out/'status.json',{'status':'confirmed' if qualified else 'stopped_confirmation_not_qualified',
                              'checkpoint':selected['checkpoint']})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(finish(args.run),indent=2),flush=True)
