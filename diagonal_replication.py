"""Prespecified wall sharding and complete-coverage assembly for fixed replication."""
import copy
from pathlib import Path

from scripts.audit_score_confirmation import audit_rows


def shard_configs(config,parts=2):
    count=config['confirmation_seeds']
    if parts<1 or count<parts or count%parts:raise ValueError('Independent walls must split into equal nonempty shards')
    shards=[]
    for index in range(parts):
        item=copy.deepcopy(config)
        item.update(out=str(Path(config['out'])/f'shard-{index+1:02d}'),
                    confirmation_seed_start=config['confirmation_seed_start']+index*(count//parts),
                    confirmation_seeds=count//parts,confirmation_wall_minutes=180)
        shards.append(item)
    return shards


def combine_rows(shards,config):
    rows=sorted((row for shard in shards for row in shard),key=lambda r:(r['seed'],r['seat']))
    audit_rows(rows,config)
    return rows
