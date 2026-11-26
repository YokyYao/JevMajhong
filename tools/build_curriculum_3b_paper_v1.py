"""Build an evidence-backed paper section and real trend figures on a GPU node.

The completed report supplies matched outcomes. Live training journals are
captured as complete-line prefixes. This never trains, invents curve points,
selects a checkpoint or opens the reserved confirmation grid.
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
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write

DATA_CSV = ROOT/'docs/experiments/results/curriculum-active3B-data-20261125.csv'
ANCESTRY = ROOT/'runs/curriculum-active-student-source-inventory-v1/snapshot-250/actual-parameter-ancestry-private.json'


def tex(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def ema(values):
    result = []; current = None
    for value in values:
        current = value if current is None else .1*value+.9*current
        result.append(current)
    return result


def build(comparison, last_step, out):
    allocation = resource()
    comparison, out = Path(comparison).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or last_step not in (1875, 2500):
        raise ValueError('Use a new private output and an actual1875/2500 report')
    binding = comparison/'completion-binding-private.json'
    bound = read(binding)['files']; files = {str(binding): sha(binding), str(Path(__file__)): sha(__file__)}
    def checked(path):
        path = Path(path).resolve(); digest = sha(path)
        if bound.get(str(path)) != digest: raise ValueError('Completed report input changed: '+str(path))
        files[str(path)] = digest
        return read(path)
    done = checked(comparison/'summary-private.json')
    journals = checked(comparison/'selected-journal-points-private.json')
    points = checked(comparison/'projected-curve16-points-private.json')
    endpoint64 = checked(comparison/'original-endpoint64-aggregates-private.json')
    steps = done['actual_steps']
    if (done['status'] != 'actual_same16_development_points_and_full_validation_summarized'
            or steps[-1] != last_step or steps != list(range(0, last_step+1, 625))
            or done['matches_per_point'] != 16 or done['consumed_seed_groups'] != 2
            or not done['same_grid_and_execution_policy']):
        raise ValueError('Require every real registered same16 observation through the requested step')
    registrations = [Path(name) for name in bound if name.endswith('/fit-plan/registered.json')]
    if len(registrations) != 1: raise ValueError('One actual comparison registration required')
    reg = checked(registrations[0]); config = reg['curriculum']
    ready = checked(ROOT/config['data_ready']); fit = ROOT/config['out']
    evaluated_model = checked(fit/'evaluations'/f'step-{last_step:06d}'/'protocol.json')['checkpoint']
    if last_step == 2500:
        if '2500' not in endpoint64 or endpoint64['2500']['matches'] != 64:
            raise ValueError('The final complete64-match endpoint must be retained separately')
        from tools.curriculum_placement_early_parent_v1 import parent_artifacts
        parent_artifacts(registrations[0])
    with DATA_CSV.open(newline='') as stream: sources = list(csv.DictReader(stream))
    if len(sources) != 25 or len({row['source'] for row in sources}) != 25:
        raise ValueError('All actual25 data sources required')
    for row in sources:
        source = row['source']; n = int(row['train_records']); mass = float(row['loss_mass'])
        if (n != ready['coverage']['train']['sources'][source]
                or int(row['validation_records']) != ready['coverage']['validation']['sources'][source]
                or not math.isclose(float(row['scalar_example_weight']), mass*91247/n, abs_tol=1e-8)):
            raise ValueError('Published mix must match actual frozen source counts and weights')
    if (sum(int(r['train_records']) for r in sources) != 91247
            or sum(int(r['validation_records']) for r in sources) != 24516
            or not math.isclose(sum(float(r['loss_mass']) for r in sources
                if r['new_3B_route'] == 'True'), .2, abs_tol=1e-9)
            or not math.isclose(sum(float(r['loss_mass']) for r in sources), 1, abs_tol=1e-9)):
        raise ValueError('The exact registered80/20 mass and complete data split are required')
    files[str(DATA_CSV)] = sha(DATA_CSV)
    prefixes = []
    def journal(path, limit, kind):
        raw = Path(path).read_bytes(); complete = raw[:raw.rfind(b'\n')+1]
        rows = [json.loads(line) for line in complete.splitlines() if line.strip()]
        rows = [row for row in rows if row['local_step'] <= limit]
        if not rows or any(not math.isfinite(row['loss']) for row in rows):
            raise ValueError('Actual finite loss observations required')
        if kind == 'train' and [row['local_step'] for row in rows] != list(range(1, limit+1)):
            raise ValueError('Every actual optimizer update through the selected phase boundary required')
        prefixes.append(dict(path=str(Path(path).resolve()), complete_prefix_bytes=len(complete),
            complete_prefix_sha256=hashlib.sha256(complete).hexdigest(), selected_step_limit=limit))
        return rows
    train = journal(fit/'metrics/train_metrics.jsonl', last_step, 'train')
    valid = [journals['validation'][str(step)] for step in steps]
    outcomes = [points[str(step)] for step in steps]
    ancestry = list(reversed(read(ANCESTRY)))
    if [row['selected_actual_saved_step'] for row in ancestry] != [608, 388, 1800, 250]:
        raise ValueError('Only actual selected basic608/calls388/3A1800 ancestry is supported')
    files[str(ANCESTRY)] = sha(ANCESTRY)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 10})
    out.mkdir(parents=True, exist_ok=False); figures = out/'figures'; figures.mkdir()
    def save(fig, name):
        for suffix in ('pdf', 'png', 'svg'): fig.savefig(figures/(name+'.'+suffix), dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(7, 7), layout='constrained')
    xs = [row['local_step'] for row in train]; ys = [row['loss'] for row in train]
    axes[0].plot(xs, ys, alpha=.16, linewidth=.5, color='#7b8799', rasterized=True, label='Actual minibatch objective')
    axes[0].plot(xs, ema(ys), color='#235fa4', linewidth=1.1, label='EMA, alpha=0.1')
    axes[0].set_title('Training objective'); axes[0].legend(fontsize=7)
    axes[1].plot(steps, [row['loss'] for row in valid], 'o-', color='#b74b30')
    axes[1].set_title('Fixed validation objective (zoomed)')
    for step, row in zip(steps, valid):
        axes[1].annotate(f"{row['loss']:.4f}", (step, row['loss']), xytext=(0, 7),
            textcoords='offset points', ha='center', fontsize=8)
    for key, label in [('loss', 'Total objective'), ('natural_target_ce', 'Natural target CE'),
                       ('new_route_objective', 'New-route contribution')]:
        baseline = valid[0][key]
        axes[2].plot(steps, [100*(row[key]/baseline-1) for row in valid], 'o-', label=label)
    axes[2].set_title('Validation change from step0 (%)'); axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel('3B optimizer updates'); ax.grid(alpha=.2); ax.margins(x=.08, y=.18)
    save(fig, 'curriculum3b_loss')
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), layout='constrained')
    for ax, (key, label, scale) in zip(axes.flat, [
            ('reward', 'Match reward (net /8000)', 1), ('hand_win_rate', 'Hand win rate (%)', 100),
            ('top2_rate', 'Top-two rate (%)', 100), ('net_points', 'Net points / match', 1)]):
        values = [row[key]*scale for row in outcomes]
        ax.plot(steps, values, 'o-', color='#235fa4')
        for x, value in zip(steps, values):
            text = f'{value:,.0f}' if key == 'net_points' else f'{value:.2f}'
            ax.annotate(text, (x, value), xytext=(0, 6), textcoords='offset points',
                ha='center', fontsize=8)
        ax.set_title(label); ax.set_xlabel('3B optimizer updates'); ax.grid(alpha=.2)
        ax.margins(x=.1, y=.22)
    save(fig, 'curriculum3b_outcomes')
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), layout='constrained')
    for index, (ax, parent, label) in enumerate(zip(axes.flat, ancestry,
            ('1: Basic / riichi', '2: Calls / post-meld', '3A: Efficiency / defense', '3B: Yaku / routes'))):
        limit = parent['selected_actual_saved_step'] if index < 3 else last_step
        folder = Path(parent['protocol']).parent/'metrics'
        tr = journal(folder/'train_metrics.jsonl', limit, 'train')
        va = journal(folder/'validation_metrics.jsonl', limit, 'validation')
        tx = [row['local_step'] for row in tr]; ty = [row['loss'] for row in tr]
        ax.plot(tx, ty, color='#8592a3', alpha=.15, linewidth=.5, rasterized=True)
        ax.plot(tx, ema(ty), color='#235fa4', linewidth=1, label='Train EMA alpha0.1')
        ax.plot([row['local_step'] for row in va], [row['loss'] for row in va],
            'o-', color='#b74b30', markersize=3, label='Validation objective')
        ax.set_title(label+'\n'+str(limit)+' selected updates'); ax.set_xlabel('Phase-local updates')
        ax.set_ylabel('Phase-specific objective'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    save(fig, 'curriculum_selected_route_loss')
    table = [r'\begin{tabular}{l'+'r'*len(steps)+'}', r'\toprule',
        'Metric & '+' & '.join(str(step) for step in steps)+r' \\', r'\midrule']
    for kind, key, label, scale in [('validation', 'loss', 'Validation objective', 1),
            ('evaluation', 'reward', 'Match reward', 1), ('evaluation', 'hand_win_rate', 'Hand win (\\%)', 100),
            ('evaluation', 'top2_rate', 'Top two (\\%)', 100), ('evaluation', 'net_points', 'Net points / match', 1),
            ('evaluation', 'first_rate', 'First place (\\%)', 100), ('evaluation', 'dealin_hand_rate', 'Deal-in (\\%)', 100)]:
        values = [row[key]*scale for row in (valid if kind == 'validation' else outcomes)]
        digits = 4 if key in ('loss', 'reward') else 2
        table.append(label+' & '+' & '.join(f'{value:.{digits}f}' for value in values)+r' \\')
    table += [r'\bottomrule', r'\end{tabular}']
    (out/'curriculum_results_table.tex').write_text('\n'.join(table)+'\n')
    endpoint_block = ('\\paragraph{Full endpoint pending.} The registered64-match final endpoint is not yet available; '
        'only completed development observations are reported here.')
    if endpoint64:
        end_table = [r'\begin{tabular}{rrrrrr}', r'\toprule',
            r'Step & Matches & Top two (\%) & Net points & Hand win (\%) & Deal-in (\%) \\', r'\midrule']
        for step, row in sorted(endpoint64.items(), key=lambda item: int(item[0])):
            end_table.append(f"{step} & {row['matches']} & {100*row['top2_rate']:.2f} & {row['net_points']:.2f} & "
                f"{100*row['hand_win_rate']:.2f} & {100*row['dealin_hand_rate']:.2f}"+r' \\')
        end_table += [r'\bottomrule', r'\end{tabular}']
        (out/'curriculum_endpoint64_table.tex').write_text('\n'.join(end_table)+'\n')
        endpoint_block = r'''\begin{table}[t]
\centering\small
\caption{Separate complete64-match development endpoint. These eight consumed
seed groups are not joined to the fixed16-match learning curve and are not the
independent final comparison against frozen v8.}
\label{tab:curriculum-endpoint64}
\input{curriculum_endpoint64_table}
\end{table}'''
    names = dict(plain='ordinary decisions', riichi_no='riichi decline', riichi_yes='riichi accept',
        call='calls', kan='kans', post_meld='after meld', post_reach_discard='post-riichi discard',
        call_window='call windows', closed_discard='closed discard', kan_window='kan windows',
        open_discard='open discard', closed_discard_under_riichi='under-riichi discard',
        ordinary_completion='ordinary completion', chiitoitsu='Chiitoitsu', chuuren='Chuuren poutou',
        daisangen='Daisangen', kokushi='Kokushi musou', suuankou='Suuankou',
        unselected_random_initial_hand_control='random-hand control')
    data_table = [r'\begin{tabular}{llrrr}', r'\toprule',
        r'Phase & Source & TRAIN & VAL & Loss mass (\%) \\', r'\midrule']
    for row in sources:
        parts = row['source'].split(':')
        phase = ('Basic' if parts[1] == 'basic' else 'Calls' if parts[1] == 'advanced'
                 else 'Post-riichi' if parts[1] == 'followup' else '3A') if parts[0] == 'curriculum2' else parts[0].replace('curriculum', '')
        if phase == '3': phase = '3A'
        role = parts[-2].replace('mc', 'simulation')
        label = role+': '+names[parts[-1]]
        data_table.append(f"{phase} & {tex(label)} & {int(row['train_records']):,} & "
            f"{int(row['validation_records']):,} & {100*float(row['loss_mass']):.4f}"+r' \\')
    data_table += [r'\midrule', r'Total & 25 sources & 91,247 & 24,516 & 100.0000 \\',
        r'\bottomrule', r'\end{tabular}']
    (out/'curriculum_data_mix.tex').write_text('\n'.join(data_table)+'\n')
    status = ('Stage 3B has completed its registered 2,500-update budget and execution audit.' if last_step == 2500
              else 'This is an interim development snapshot through step 1,875; the registered 2,500-update Stage 3B run is still in progress.')
    section = r'''% Generated from actual completed evidence by build_curriculum_3b_paper_v1.py.
\section{Pinned-Base Curriculum Reproduction}\label{sec:curriculum-reproduction}
STATUS
This experiment is separate from the historical released-policy comparisons above.
Its parameter ancestry starts at \texttt{Qwen/Qwen3.5-0.8B}, pinned to revision
\texttt{2fc06364715b967f1860aea9cf38778875588b17}; frozen v8 is an evaluation reference,
not an initialization checkpoint. The selected ancestry is basic discard/riichi
(608 updates), calls and post-meld play (388), efficiency/defense (3A, 1,800),
and yaku/route learning (3B, 2,500 planned). A subsequent placement stage is not
claimed complete here. The 3A parent was chosen from ten historical checkpoints
on consumed development games; this is a documented adaptive selection, not an
independent test result or one uninterrupted optimization run.

\paragraph{Data and objective.}
Stage 3B retains 91,247 TRAIN and 24,516 VAL decision menus across 25 sources.
Nineteen replay sources receive 80\% of the objective mass; six new sources
receive 20\%, equally divided across Kokushi musou, Chuuren poutou, Suuankou,
Daisangen, Chiitoitsu, and a random-initial-hand control. These are generation
targets and teacher-label strata, not guarantees that a trajectory completes
the named yaku. The new sources contain 3,652 TRAIN and 828 VAL menus; the
validation menus cover 45 source families. Repeated decisions from one family
are not independent matches. Source-group splits and inherited replay are preserved.
Table~\ref{tab:curriculum-data} reports the complete mix: record frequency and
objective mass are distinct. For source $s$ with $n_s$ training records and target
mass $m_s$, the scalar example weight is $w_s=m_s\,91247/n_s$.

The frozen backbone uses a shared rank-16 LoRA adapter and decision head.
The registered recipe uses two epochs (182,494 record exposures), batch size one,
gradient accumulation 73, adapter learning rate $3\!\times\!10^{-6}$ and head
learning rate $1.5\!\times\!10^{-5}$. Initialization retains the parent adapter/head
and creates a fresh optimizer. Each epoch uses a deterministic source-mixed
ordering with mixing blocks of 511 records. This is replay-based supervised
fine-tuning, including retained simulation-derived soft targets; evaluated match
reward is not an online reinforcement-learning training signal.

\paragraph{Actual learning curves and development outcomes.}
The complete validation set is evaluated at 0, 625, 1,250, 1,875 and 2,500 updates.
Figures~\ref{fig:curriculum-loss}--\ref{fig:curriculum-outcomes} contain only completed
observations. Minibatch training objectives and their EMA are shown separately
from fixed-set validation loss. Validation changes are also expressed relative
to step zero to expose small genuine trends. Axis ranges are fitted to observed
values and clearly labeled; no curve points are synthesized or fitted to imply
an improvement. A flat segment remains flat when the recorded metric is unchanged.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/curriculum3b_loss.pdf}
\caption{Actual 3B training objective (raw observations and EMA with $\alpha=0.1$),
fixed validation objective with a zoomed vertical axis, and relative validation
changes. The new-route value is a weighted objective contribution, not a win rate.
Only the training objective is smoothed.}\label{fig:curriculum-loss}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/curriculum3b_outcomes.pdf}
\caption{Realized outcomes on the same 16 complete East matches at every point:
two consumed logical seeds, two opponent profiles and four seats. Lines connect
recorded observations only; the small, correlated grid does not establish stable
improvement or superiority to v8. Reward equals terminal net points divided by
8,000; hand-win rate and placement rates have different denominators.}
\label{fig:curriculum-outcomes}
\end{figure}

\begin{table}[t]
\centering\small
\caption{Completed same-grid development observations. Columns are local 3B
optimizer updates. These values are not the independent final comparison.}
\label{tab:curriculum-development}
\input{curriculum_results_table}
\end{table}

ENDPOINT64TABLE

The fixed development curve uses 16 matches throughout. At the final endpoint,
all 64 registered development matches are also reported separately; its
preregistered first-two-seed subset supplies the comparable 16-match curve.
This prevents a grid change from appearing as a training improvement.
The independently reserved confirmation remains unopened: 32 logical seeds,
two opponent profiles and four seats give 256 complete East matches per policy.
The final post-placement candidate must be frozen before that comparison with v8;
current development results do not establish parity.

\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/curriculum_selected_route_loss.pdf}
\caption{Actual selected parameter ancestry through the reported 3B step.
Each panel has its own phase-local update axis and phase-specific objective;
different data mixtures and objectives are not joined into a single comparable
loss series. Training EMA uses $\alpha=0.1$; validation observations are unsmoothed.}
\label{fig:curriculum-route-loss}
\end{figure}

\begin{table}[t]
\centering\scriptsize
\caption{Complete frozen 3B source mix. Loss mass is a prescribed objective
weight, not a measured gradient-contribution fraction or a record-frequency fraction.}
\label{tab:curriculum-data}
\input{curriculum_data_mix}
\end{table}
'''.replace('STATUS', status).replace('ENDPOINT64TABLE', endpoint_block)
    (out/'curriculum_reproduction.tex').write_text(section)
    write(out/'journal-prefix-receipts-private.json', prefixes)
    write(out/'summary-private.json', dict(status='actual_curriculum_paper_material_built',
        last_step=last_step, actual_steps=steps, actual_final2500_audited=last_step == 2500,
        data_sources=25, train_records=91247, validation_records=24516,
        replay_objective_mass=.8, new_route_objective_mass=.2,
        figures_use_real_observations=True, artificial_trends=False,
        actual_selected_ancestry_steps=[608,388,1800,last_step],
        evaluated_parameter_identity=evaluated_model['identity_sha256'],
        new_games=0, model_loaded=False, student_updates=0, allocation=allocation))
    for path in out.rglob('*'):
        if path.is_file(): files[str(path)] = sha(path)
    write(out/'completion-binding-private.json', dict(files=files))
    print(json.dumps(dict(status='actual_paper_material_built', last_step=last_step,
        sources=25, final2500_audited=last_step == 2500, new_games=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--last-step', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); build(args.comparison, args.last_step, args.out)
