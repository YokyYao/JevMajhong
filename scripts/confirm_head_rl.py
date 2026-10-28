"""One frozen candidate, fresh confirmation seeds, no further training or selection."""
import json
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena import ci
from head_rl import summarize, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'runs/head-rl-pilot-v1')
    args = parser.parse_args()
    out = args.run
    selection = json.loads((out/'confirmation-selection.json').read_text())
    config = selection['config']
    data = {}
    for name, checkpoint in [('reference', config['reference']), ('candidate', selection['checkpoint'])]:
        directory = out/('confirmation-'+name)
        command = [sys.executable, str(ROOT/'arena.py'), '--checkpoint', checkpoint, '--guarded',
                   '--seeds', str(config['confirmation_seeds']), '--seed-start', str(config['confirmation_seed_start']),
                   '--seed-scheme', config.get('seed_scheme', 'spaced-v1'),
                   '--out', str(directory)]
        if directory.exists():
            command.append('--resume')
        subprocess.run(command, cwd=ROOT, check=True)
        rows = [json.loads(s)['candidate'] for s in (directory/'games.jsonl').read_text().splitlines()]
        expected = {(s, seat) for s in range(config['confirmation_seed_start'],
                                              config['confirmation_seed_start']+config['confirmation_seeds']) for seat in range(4)}
        if len(rows) != len(expected) or {(r['seed'], r['seat']) for r in rows} != expected:
            raise ValueError('Incomplete confirmation coverage')
        data[name] = rows
    from scripts.evaluate_round2 import paired_comparison
    paired = paired_comparison(data['candidate'], data['reference'])
    a = {(r['seed'], r['seat']): r for r in data['candidate']}
    b = {(r['seed'], r['seat']): r for r in data['reference']}
    seeds = sorted({s for s, _ in a})
    first_ci = ci([sum(int(a[s, seat]['rank'] == 1)-int(b[s, seat]['rank'] == 1) for seat in range(4))/4
                   for s in seeds], confidence=.95)
    candidate, reference = summarize(data['candidate']), summarize(data['reference'])
    qualified = (candidate['first_rate'] >= config['target_first_rate']
                 and candidate['first_rate']-reference['first_rate'] >= config['target_first_gain']
                 and first_ci[0] > 0 and candidate['mean_net_points'] >= reference['mean_net_points']
                 and candidate['fourth_rate'] <= reference['fourth_rate']+config['target_fourth_tolerance'])
    result = {'status': 'complete', 'qualified': qualified, 'candidate': candidate, 'reference': reference,
              'first_delta_ci95': first_ci, 'paired': paired, 'checkpoint': selection['checkpoint'],
              'default_policy_changed': False, 'note': 'One preselected candidate, 128 independent seeds; no training on confirmation seeds.'}
    write(out/'confirmation-result.json', result)
    write(out/'status.json', {'status': 'confirmed' if qualified else 'stopped_confirmation_not_qualified',
                             'checkpoint': selection['checkpoint']})
    with (out/'analysis.md').open('a') as stream:
        stream.write('\n## 独立确认结果\n\n')
        stream.write(f"候选一位率 {candidate['first_rate']:.2%}，旧模型 {reference['first_rate']:.2%}；"
                     f"配对一位率差95%区间 {first_ci}。候选平均净得点 {candidate['mean_net_points']:.1f}，"
                     f"旧模型 {reference['mean_net_points']:.1f}。阶段目标{'通过' if qualified else '未通过，停止此配方'}。\n")
    print('confirmation_result '+json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
