"""Freeze one positive existing head for one larger fresh, fixed-sample replication."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from diagonal_replication import shard_configs
from scripts.audit_score_confirmation import audit,require,sha256


def write(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def prepare(source,out):
    require(not out.exists(),'Preserve existing partial or completed replication')
    old=json.loads((source/'protocol.json').read_text());chosen=json.loads((source/'confirmation-selection.json').read_text())
    expected=chosen['candidate_head_sha256'];checked=audit(source,expected)
    require(checked['primary_delta']>0,'The fixed source must have an actual positive paired score estimate')
    smoke=json.loads((source/'recommendation-interface-smoke.json').read_text())
    require(smoke['status']=='passed' and smoke['head_sha256']==expected and smoke['source_checkpoint_unchanged'],
            'Missing matching usable-interface evidence')
    require(all(smoke.get(name) is True for name in ['real_player_observation','legal_guarded_action_returned',
                'public_cli_exercised','matches_evaluated_head_policy_action']),
            'Interface evidence must exercise a real legal matching public recommendation')
    for name,sha in smoke['code_sha256'].items():require(sha256(ROOT/name)==sha,'Recommendation runtime changed')
    config={**old['config'],'out':str(out),'confirmation_seed_start':4602000,'confirmation_seeds':512,
            'confirmation_wall_minutes':360,'target_mean_net_points':0,'target_net_points_gain':0,
            'fixed_gain_gate_enabled':False}
    require(old['reference_head_sha256']=='3c586e1bf6e86892c29a1f76bbab4fda4f714b17a31f87c6fe871157748a7d5b',
            'Keep the original SFT2000 benchmark')
    proposed=set(range(config['confirmation_seed_start'],config['confirmation_seed_start']+config['confirmation_seeds']))
    for path in (ROOT/'runs').glob('*/protocol.json'):
        previous=json.loads(path.read_text()).get('config',{})
        for start,length in [('training_seed_start','training_seeds_per_stage'),('development_seed_start','development_seeds'),
                             ('confirmation_seed_start','confirmation_seeds')]:
            if start in previous and length in previous:
                count=previous.get('max_stages',1) if start=='training_seed_start' else 1
                require(not proposed&set(range(previous[start],previous[start]+previous[length]*count)),
                        f'Fresh replication walls already declared:{path}')
    code={**old['code_sha256'],**smoke['code_sha256'],**{name:sha256(ROOT/name) for name in
          ['diagonal_replication.py','scripts/prepare_diagonal_replication.py','scripts/finish_diagonal_replication.py']}}
    common={**old,'source_run':str(source),'source_protocol_sha256':old['sha256'],
            'code_tree_sha':subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip(),
            'code_sha256':code,'study_type':'fixed_preserved_positive_head_independent_larger_replication',
            'shards_predeclared':2,'combine_all_shards_before_primary_decision':True,
            'source_confirmation_not_relabelled':True}
    training=sorted(source.glob('training-games-stage-*.json'));require(bool(training),'Missing training provenance')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent,prefix=out.name+'.prepare-') as temporary:
        staged=Path(temporary)/'study';staged.mkdir()
        for index,cfg in enumerate([config,*shard_configs(config)]):
            directory=staged if index==0 else staged/f'shard-{index:02d}';directory.mkdir(exist_ok=True)
            protocol={**common,'config':cfg,'sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()}
            selected={**chosen,'config':cfg,'selection_rule':'same preserved positive head, no reselection after source confirmation',
                      'source_run':str(source),'candidate_head_sha256':expected}
            write(directory/'protocol.json',protocol);write(directory/'confirmation-selection.json',selected)
            write(directory/'status.json',{'status':'awaiting_fixed_replication'})
            for path in training:shutil.copy2(path,directory/path.name)
        write(staged/'source-confirmation-audit.json',checked)
        shutil.copy2(source/'confirmation-result.json',staged/'source-confirmation-result.json')
        shutil.copy2(source/'recommendation-interface-smoke.json',staged/'source-recommendation-interface-smoke.json')
        staged.rename(out)
    return {'out':str(out),'fixed_candidate_head_sha256':expected,'new_independent_streams':512,
            'games_per_policy':2048,'shards':2,'source_qualification_unchanged':True}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(prepare(args.source,args.out)),flush=True)
