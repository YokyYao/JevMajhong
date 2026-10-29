"""Freeze the best positive completed source checkpoint before fresh-data training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from head_rl import fingerprint,write
from incremental_score import choose_positive,verified_development


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--config-out',type=Path,required=True)
    args=parser.parse_args()
    if args.config_out.exists():
        raise ValueError('Preserve the existing prepared configuration')
    protocol,baseline,results=verified_development(args.source)
    selected=choose_positive(results,baseline['mean_net_points'])
    if selected is None:
        raise ValueError('No positive source checkpoint; analyze rather than discard the original model')
    config=json.loads(args.template.read_text())
    new_ranges={}
    for start,length,count in [('training_seed_start','training_seeds_per_stage',config['max_stages']),
                               ('development_seed_start','development_seeds',1),
                               ('confirmation_seed_start','confirmation_seeds',1)]:
        new_ranges[start]=set(range(config[start],config[start]+config[length]*count))
    keys=list(new_ranges)
    if any(new_ranges[a]&new_ranges[b] for i,a in enumerate(keys) for b in keys[i+1:]):
        raise ValueError('New training/development/confirmation ranges overlap')
    all_new=set().union(*new_ranges.values())
    for path in (ROOT/'runs').glob('*/protocol.json'):
        previous=json.loads(path.read_text()).get('config',{})
        for start,length in [('training_seed_start','training_seeds_per_stage'),
                             ('development_seed_start','development_seeds'),
                             ('confirmation_seed_start','confirmation_seeds')]:
            if start in previous and length in previous:
                count=previous.get('max_stages',1) if start=='training_seed_start' else 1
                if all_new&set(range(previous[start],previous[start]+previous[length]*count)):
                    raise ValueError(f'New stream range already declared by {path}')
    checkpoint=ROOT/selected['checkpoint']
    config.update(reference=selected['checkpoint'],initial_reference_head_sha256=fingerprint(checkpoint/'head.pt'),
                  initialization_source_run=str(args.source),initialization_source_update=selected['update'],
                  initialization_source_protocol_sha256=protocol['sha256'],
                  initialization_selection_rule='highest complete development net points above source reference; earliest tie')
    write(args.config_out,config)
    print(json.dumps({'config':str(args.config_out),'reference':config['reference'],
                      'head_sha256':config['initial_reference_head_sha256'],
                      'source_development_mean':selected['metrics']['mean_net_points'],
                      'fixed_gain_continuation_gate':False}),flush=True)


if __name__=='__main__':
    main()
