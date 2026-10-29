"""One frozen candidate, fresh confirmation seeds, no further training or selection."""
import json
import argparse
import copy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena import ci
from head_rl import HeadPolicy, paired_primary_comparison, primary_metric, summarize, write
from scripts.train_head_rl import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'runs/head-rl-pilot-v1')
    args = parser.parse_args()
    out = args.run
    selection = json.loads((out/'confirmation-selection.json').read_text())
    config = selection['config']
    if (out/'confirmation-result.json').exists():
        print((out/'confirmation-result.json').read_text(), flush=True)
        return
    # Use the same unmerged native runtime and guarded greedy path as development.
    # Keep the checkpoint frozen and reuse its unchanged backbone for both heads.
    policy = HeadPolicy(config, selection['checkpoint'])
    confirmation_config = copy.deepcopy(config)
    confirmation_config['development_seed_start'] = config['confirmation_seed_start']
    confirmation_config['development_seeds'] = config['confirmation_seeds']
    deadline = time.monotonic()+config.get('confirmation_wall_minutes', 55)*60
    write(out/'status.json', {'status': 'confirmation_validation', 'checkpoint': selection['checkpoint']})
    data = {}
    for name in ['reference', 'candidate']:
        directory = out/('confirmation-'+name)
        _, rows = evaluate(policy, directory, confirmation_config, deadline, frozen=name == 'reference')
        expected = {(s, seat) for s in range(config['confirmation_seed_start'],
                                              config['confirmation_seed_start']+config['confirmation_seeds']) for seat in range(4)}
        if len(rows) != len(expected) or {(r['seed'], r['seat']) for r in rows} != expected:
            raise ValueError('Incomplete confirmation coverage')
        data[name] = rows
    paired = paired_primary_comparison(data['candidate'], data['reference'], config)
    a = {(r['seed'], r['seat']): r for r in data['candidate']}
    b = {(r['seed'], r['seat']): r for r in data['reference']}
    seeds = sorted({s for s, _ in a})
    first_ci = ci([sum(int(a[s, seat]['rank'] == 1)-int(b[s, seat]['rank'] == 1) for seat in range(4))/4
                   for s in seeds], confidence=.95)
    candidate, reference = summarize(data['candidate']), summarize(data['reference'])
    metric = primary_metric(config)
    rate_key = 'target_top2_rate' if metric == 'top2_rate' else 'target_first_rate'
    gain_key = 'target_top2_gain' if metric == 'top2_rate' else 'target_first_gain'
    primary_ci = paired['primary_delta_ci95']
    qualified = (candidate[metric] >= config[rate_key]
                 and candidate[metric]-reference[metric] >= config[gain_key]
                 and primary_ci is not None and primary_ci[0] > 0 and candidate['mean_net_points'] >= reference['mean_net_points']
                 and candidate['fourth_rate'] <= reference['fourth_rate']+config['target_fourth_tolerance'])
    result = {'status': 'complete', 'qualified': qualified, 'candidate': candidate, 'reference': reference,
              'primary_metric': metric, 'primary_delta_ci95': primary_ci,
              'game_mode': config.get('game_mode', '4p-red-half'),
              'first_delta_ci95': first_ci, 'paired': paired, 'checkpoint': selection['checkpoint'],
              'default_policy_changed': False, 'note': f"One preselected candidate, {config['confirmation_seeds']} independent seeds; no training on confirmation seeds."}
    write(out/'confirmation-result.json', result)
    write(out/'status.json', {'status': 'confirmed' if qualified else 'stopped_confirmation_not_qualified',
                             'checkpoint': selection['checkpoint']})
    with (out/'analysis.md').open('a') as stream:
        stream.write('\n## 独立确认结果\n\n')
        stream.write(f"主要指标 {metric}：候选 {candidate[metric]:.2%}，旧模型 {reference[metric]:.2%}；"
                     f"配对主要指标差95%区间 {primary_ci}。候选平均净得点 {candidate['mean_net_points']:.1f}，"
                     f"旧模型 {reference['mean_net_points']:.1f}。阶段目标{'通过' if qualified else '未通过，停止此配方'}。\n")
    print('confirmation_result '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
