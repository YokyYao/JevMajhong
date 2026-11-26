"""Draw real loss and same-grid match curves from an already completed report.

No inference, new games, optimization or checkpoint selection. Execute only in
a finite allocated GPU job. Raw journals and receipts remain private in runs/.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha


def plot(report, journal, last_step, out):
    allocation = resource()
    report, journal, out = (Path(p).resolve() for p in (report, journal, out))
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a new private runs/ directory')
    summary_path = report/'summary-private.json'
    binding_path = report/'completion-binding-private.json'
    summary = read(summary_path)
    bound = read(binding_path)['files']
    inputs = [summary_path, report/'aggregate-comparison.csv',
              report/'projected-curve16-points-private.json']
    if any(bound.get(str(path)) != sha(path) for path in inputs):
        raise ValueError('Actual completed report inputs must retain their original receipt')
    steps = summary['actual_steps']
    if (summary['status'] != 'actual_same16_development_points_and_full_validation_summarized'
            or summary['matches_per_point'] != 16 or summary['consumed_seed_groups'] != 2
            or not summary['same_grid_and_execution_policy']
            or not summary['endpoint64_not_joined_to_curve']
            or steps != sorted(set(steps)) or steps[0] != 0 or steps[-1] != last_step):
        raise ValueError('Require completed same16 development points through the requested step')
    points = read(report/'projected-curve16-points-private.json')
    with (report/'aggregate-comparison.csv').open(newline='') as stream:
        comparison = list(csv.DictReader(stream))
    series = {}
    for row in comparison:
        key = (row['kind'], row['metric'])
        step, value = int(row['step']), float(row['value'])
        if not math.isfinite(value) or step not in steps or step in series.setdefault(key, {}):
            raise ValueError('Every aggregate must have one finite observation per actual step')
        series[key][step] = value
        if row['kind'] == 'evaluation' and not math.isclose(
                value, points[str(step)][row['metric']], rel_tol=0, abs_tol=1e-10):
            raise ValueError('Curve values must match the explicit same16 projection')
    if any(sorted(values) != steps for values in series.values()):
        raise ValueError('No missing milestones may be filled or interpolated')
    raw = journal.read_bytes()
    complete = raw[:raw.rfind(b'\n')+1]
    train = [json.loads(line) for line in complete.splitlines() if line.strip()]
    train = [row for row in train if row['local_step'] <= last_step]
    if ([row['local_step'] for row in train] != list(range(1, last_step+1))
            or any(not math.isfinite(row['loss']) for row in train)
            or len({(row['run_id'], row['stage']) for row in train}) != 1):
        raise ValueError('Require every actual optimizer update in one run through the requested step')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    xs = [row['local_step'] for row in train]
    ys = [row['loss'] for row in train]
    smooth = []
    current = None
    for value in ys:
        current = value if current is None else .1*value+.9*current
        smooth.append(current)
    ax = axes.flat[0]
    ax.plot(xs, ys, color='#8491a5', alpha=.25, linewidth=.6, label='Actual batch objective')
    ax.plot(xs, smooth, color='#2864b7', label='Objective EMA (alpha 0.1)')
    valid = series['validation', 'loss']
    ax.plot(steps, [valid[step] for step in steps], 'o-', color='#cc553d', label='Validation objective')
    ax.set_title('Training and validation loss')
    ax.legend(fontsize=8)
    metrics = [('reward', 'Match reward (net points / 8000)', 1),
               ('hand_win_rate', 'Hand win rate (%)', 100),
               ('top2_rate', 'Top-two rate (%)', 100),
               ('net_points', 'Net points / match', 1)]
    for ax, (metric, title, scale) in zip(list(axes.flat)[1:], metrics):
        values = series['evaluation', metric]
        ax.plot(steps, [values[step]*scale for step in steps], 'o-', color='#2864b7')
        ax.set_title(title)
    for ax in list(axes.flat)[:5]:
        ax.set_xlabel('Local optimizer updates')
        ax.grid(alpha=.22)
    axes.flat[5].axis('off')
    axes.flat[5].text(0, .95,
        'Same 16 completed East matches at every point.\n'
        'Only two consumed seed groups: descriptive,\n'
        'not independent evidence of strength.\n\n'
        'Full endpoint64 results stay in the report,\n'
        'separate from these same16 curves.\n\n'
        'Match reward is an evaluation outcome;\n'
        'it is not an RL training-reward trace.\n'
        'Lines connect observed milestones only.', va='top', fontsize=10)
    fig.suptitle(f"Actual 3B development curves through step {last_step}")
    out.mkdir(parents=True, exist_ok=False)
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(out/f'development-curves.{suffix}', dpi=150)
    plt.close(fig)
    with (out/'train-loss.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['local_step', 'objective', 'objective_ema_alpha_0_1'])
        writer.writeheader()
        writer.writerows(dict(local_step=row['local_step'], objective=row['loss'],
            objective_ema_alpha_0_1=ema) for row, ema in zip(train, smooth))
    write(out/'summary-private.json', dict(
        status='actual_loss_and_same16_match_outcome_curves_exported',
        last_step=last_step, actual_evaluation_steps=steps, actual_train_updates=len(train),
        consumed_seed_groups=2, matches_per_point=16, independent_strength_claim=False,
        full_endpoint64_not_joined=True, new_games=0, model_loaded=False, student_updates=0,
        journal_complete_prefix_sha256=hashlib.sha256(complete).hexdigest(),
        journal_complete_prefix_bytes=len(complete), journal_path=str(journal), allocation=allocation))
    files = {str(path): sha(path) for path in [Path(__file__), binding_path, *inputs, *out.iterdir()]}
    write(out/'completion-binding-private.json', dict(files=files,
        active_journal_bound_by_complete_prefix_receipt=True))
    print(json.dumps(dict(status='actual_development_curves_exported', last_step=last_step,
        actual_train_updates=len(train), actual_evaluation_steps=steps, new_games=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--last-step', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plot(args.report, args.journal, args.last_step, args.out)
