"""Plot completed matched16 net-point curves and the separate final64 grid."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_progression_runtime_v1 import require_registration
from tools.plot_curriculum_fixed_grid import aligned_record


def render(registered, out):
    allocation = resource(); reg = read(registered); require_registration(reg)
    fit = ROOT/reg['curriculum']['out']; metrics = fit/'metrics'
    done = read(fit/'completion.json')
    if done['optimizer_steps'] != 2400 or done['actual_records_seen'] != 175190:
        raise ValueError('Completed full progression budget required')
    journal = metrics/'evaluation_metrics.jsonl'
    rows = [json.loads(line) for line in journal.open() if line.strip()]
    if [r['local_step'] for r in rows] != [0, 600, 1200, 1800, 2400]:
        raise ValueError('Every original completed evaluation point required')
    aligned = [aligned_record(r) for r in rows]
    if any(r['global_step'] != 996+r['local_step'] for r in aligned):
        raise ValueError('Original parameter-ancestry updates required')
    refs = {name: read(fit/'evaluations'/name/'summary.json') for name in
            ('reference-parent', 'reference-original-guarded', 'reference-v8')}
    if (refs['reference-v8']['checkpoint_sha256'] != reg['curriculum']['reference_identity']
            or any(refs[k]['checkpoint_sha256'] != reg['curriculum']['parent_identity']
                   for k in ('reference-parent', 'reference-original-guarded'))):
        raise ValueError('Exact frozen reference model summaries required')
    if any(r['matches'] != 64 or r['seed_groups'] != 8 for r in [rows[-1], *refs.values()]):
        raise ValueError('Separate final64 comparisons require the same original eight seed groups')
    if any(refs[k]['curve_grid_subset']['matches'] != 16 for k in refs):
        raise ValueError('Reference baselines need their actual matched16 subset')
    inputs = [Path(__file__), registered, fit/'completion.json', journal,
        ROOT/'tools/plot_curriculum_fixed_grid.py', metrics/'plot-data-provenance.json']
    inputs.extend(fit/'evaluations'/name/'summary.json' for name in refs)
    files = {str(p.resolve()): sha(p) for p in inputs}
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    out.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = [r['global_step'] for r in aligned]; y = [r['net_points'] for r in aligned]
    axes[0].plot(x, y, marker='o', label='Paired-order 3A, matched16')
    axes[0].axhline(refs['reference-v8']['curve_grid_subset']['net_points'], color='tab:orange',
        linestyle='--', label='Frozen v8, same16')
    axes[0].axhline(refs['reference-parent']['curve_grid_subset']['net_points'], color='grey',
        linestyle=':', label='Calls parent, same16 / full legal')
    axes[0].set(title='Development trajectory: same16 matches / two seeds',
        xlabel='Cumulative actual optimizer updates', ylabel='Mean net points per complete East match')
    axes[0].set_xticks(x, [str(n) for n in x]); axes[0].legend(fontsize=8)
    names = ['Calls parent\nfull legal', 'Calls parent\nguarded', 'Frozen v8', 'Paired3A\nfinal2400']
    values = [refs['reference-parent']['net_points'], refs['reference-original-guarded']['net_points'],
        refs['reference-v8']['net_points'], rows[-1]['net_points']]
    bars = axes[1].bar(names, values, color=['grey', 'silver', 'tab:orange', 'tab:blue'])
    axes[1].bar_label(bars, labels=[f'{v:+,.2f}' for v in values], padding=4, fontsize=9)
    axes[1].set(title='Final comparison: same64 matches / eight seeds',
        ylabel='Mean net points per complete East match')
    axes[1].set_ylim(min(0, min(values))*1.2, max(values)*1.22)
    for axis in axes:
        axis.axhline(0, color='black', linewidth=.6)
        axis.grid(axis='y', alpha=.25); axis.set_axisbelow(True)
    figure.suptitle('Consumed development outcomes; no independent strength claim', fontsize=13)
    figure.savefig(out/'net-points.png', dpi=180)
    figure.savefig(out/'net-points.pdf')
    plt.close(figure)
    if any(sha(p) != h for p, h in files.items()):
        raise ValueError('Plot input changed during rendering')
    result = dict(status='actual_completed_fixed_grid_net_points_plotted_not_confirmation',
        curve_local_steps=[r['local_step'] for r in aligned], curve_global_steps=x,
        curve_matches_per_point=16, curve_logical_seed_groups=2, curve_net_points=y,
        final_comparison_matches_per_policy=64, final_comparison_seed_groups=8,
        final_net_points=dict(zip(names, values)), original_64_not_connected_to_16_curve=True,
        new_games_model_forwards_or_optimizer_updates=0,
        independent_execution_audit_and_final_confirmation_not_replaced=True)
    write(out/'aggregate-provenance.json', result)
    for p in (out/'net-points.png', out/'net-points.pdf', out/'aggregate-provenance.json'):
        files[str(p.resolve())] = sha(p)
    write(out/'completion-binding-private.json', dict(files=files, allocation=allocation))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registered', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); render(args.registered.resolve(), args.out.resolve())
