"""Add matched64 reference results without copying the exclusive output summary.

Consumes the completed native/optimizer audit and existing aggregate results.
No models, simulations, new statistics, or reserved confirmation games.
"""
import argparse
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha

REGISTRATION = ROOT/'runs/curriculum-yaku-early-v2/fit-plan/registered.json'
REFERENCE = '5ebce6af3777854cbec55ba29db2dc90be0f959193114bb3154179c5552370fa'
OLD_CAPTION = ('Separate complete64-match development endpoint. These eight consumed\n'
    'seed groups are not joined to the fixed16-match learning curve and are not the\n'
    'independent final comparison against frozen v8.')
NEW_CAPTION = ('Matched complete64-match development results for the 3A parent, frozen v8,\n'
    'and the 3B endpoint. All policies use the same eight logical seeds, two opponent\n'
    'profiles and four seats. These consumed development results are reported\n'
    'separately from the fixed16 learning curve and the unopened final confirmation.')


def attach(material, out):
    allocation = resource()
    material, out = Path(material).resolve(), Path(out).resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a new private paper-material version')
    binding_path = material/'completion-binding-private.json'
    bound = read(binding_path)['files']
    paths = [path for path in material.rglob('*') if path.is_file()
        and path.name != 'completion-binding-private.json']
    for path in paths:
        if bound.get(str(path)) != sha(path):
            raise ValueError('Completed paper material changed: '+str(path))
    summary = read(material/'summary-private.json')
    if (summary['last_step'] != 2500 or summary['actual_final2500_audited'] is not True
            or summary.get('checkpoint_card_attached') is not True
            or summary.get('actual_packaged_load_query_passed') is not True):
        raise ValueError('Actual audited2500 material and usable checkpoint card required')
    from tools.curriculum_placement_early_parent_v1 import parent_artifacts
    parent_artifacts(REGISTRATION)
    reg = read(REGISTRATION)
    fit = ROOT/reg['curriculum']['out']
    audit_path = fit.parent/'final-evidence-private.json'
    audit = read(audit_path)
    if (audit['status'] != 'yaku2500_final_evidence_audit_passed'
            or audit['optimizer_steps'] != 2500 or audit['actual_records_seen'] != 182494
            or audit['final_identity'] != summary['evaluated_parameter_identity']
            or not audit['source_and_data_hashes_unchanged']
            or not audit['source_split_replay_and_exposure_exact']):
        raise ValueError('Actual completed native and parameter audit required')
    specs = [('reference-parent', '3A parent (1,800)', reg['curriculum']['parent_identity']),
             ('reference-v8', 'Frozen v8', REFERENCE),
             ('step-002500', '3B (2,500)', audit['final_identity'])]
    rows = {}; inputs = {str(REGISTRATION):sha(REGISTRATION), str(audit_path):sha(audit_path)}
    for label, display, identity in specs:
        folder = fit/'evaluations'/label
        row = read(folder/'summary.json')
        if (row['matches'] != 64 or row['seed_groups'] != 8
                or row['checkpoint_sha256'] != identity
                or any(not math.isfinite(row[key]) for key in
                    ('top2_rate', 'net_points', 'hand_win_rate', 'dealin_hand_rate'))):
            raise ValueError('Every actual matched64 reference aggregate is required')
        if label == 'step-002500':
            if (read(folder/'status.json') != dict(status='complete', games=64)
                    or audit['final_summary']['top2_rate'] != row['top2_rate']
                    or audit['final_summary']['net_points'] != row['net_points']):
                raise ValueError('The completed native endpoint differs from its audit')
        rows[label] = row
        for name in ('summary.json', 'protocol.json'):
            path = folder/name; inputs[str(path)] = sha(path)
    candidate = rows['step-002500']
    intervals = []
    for label, display, _ in specs[:2]:
        comparison = audit['comparisons'][label]
        for field, key in [('top2', 'top2_rate'), ('net_points', 'net_points')]:
            actual = comparison[field]
            if (actual['seed_groups'] != 8 or len(actual['ci95']) != 2
                    or any(not math.isfinite(x) for x in [actual['gain'], *actual['ci95']])
                    or abs(actual['gain']-(candidate[key]-rows[label][key])) > 1e-8):
                raise ValueError('Use the actual audited paired cluster comparison')
        intervals.append((display, comparison))
    section = (material/'curriculum_reproduction.tex').read_text()
    if section.count(OLD_CAPTION) != 1:
        raise ValueError('Review the exact original matched64 caption site')
    table = [r'\begin{tabular}{lrrrrr}', r'\toprule',
        r'Policy & Matches & Top two (\%) & Net points & Hand win (\%) & Deal-in (\%) \\',
        r'\midrule']
    for label, display, _ in specs:
        row = rows[label]
        table.append(f"{display} & 64 & {100*row['top2_rate']:.2f} & {row['net_points']:.2f} & "
            f"{100*row['hand_win_rate']:.2f} & {100*row['dealin_hand_rate']:.2f}"+r' \\')
    table += [r'\bottomrule', r'\end{tabular}']
    interval_table = [r'\begin{table}[t]', r'\centering\small',
        r'\caption{Audited paired development differences: 3B minus each reference.',
        r'Intervals are the existing two-sided 95\% $t$ intervals over eight consumed',
        r'logical-seed clusters. These are not the independently reserved final test.}',
        r'\label{tab:curriculum-reference64-differences}',
        r'\begin{tabular}{lrr}', r'\toprule',
        r'Reference & Top-two gain (pp), 95\% CI & Net-point gain, 95\% CI \\', r'\midrule']
    for display, comparison in intervals:
        top, net = comparison['top2'], comparison['net_points']
        interval_table.append(f"{display} & {100*top['gain']:+.3f} "
            f"[{100*top['ci95'][0]:+.3f}, {100*top['ci95'][1]:+.3f}] & "
            f"{net['gain']:+.2f} [{net['ci95'][0]:+.2f}, {net['ci95'][1]:+.2f}]"+r' \\')
    interval_table += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    top = audit['comparisons']['reference-v8']['top2']['gain']
    net = audit['comparisons']['reference-v8']['net_points']['gain']
    direction = ('Both primary point estimates are below frozen v8.' if top < 0 and net < 0
        else 'The two primary point estimates must be interpreted jointly with their paired intervals.')
    paragraph = (r'\paragraph{Matched 64-match reference comparison.}'+'\n'
        +f"The completed endpoint has {100*candidate['top2_rate']:.2f}\\% top-two rate and "
        +f"{candidate['net_points']:+.2f} net points per match. Relative to frozen v8, "
        +f"the paired point differences are {100*top:+.3f} percentage points and {net:+.2f} points. "
        +direction+' The earlier 93.75\\% development-curve peak is based on only 16 matches; '
        +'it is neither a 64-match result nor evidence of independent parity. '
        +'The final 3B endpoint follows the registered budget; it was not chosen by the smallest loss. '
        +'Stage 4 placement training and the independently reserved comparison remain unfinished.\n\n')
    section = section.replace(OLD_CAPTION, NEW_CAPTION)
    section += '\n'+paragraph+'\n'.join(interval_table)+'\n'
    out.mkdir(parents=True)
    for path in paths:
        # The exclusive writer below creates the augmented summary once.
        if path == material/'summary-private.json': continue
        target = out/path.relative_to(material)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (out/'curriculum_endpoint64_table.tex').write_text('\n'.join(table)+'\n')
    (out/'curriculum_reproduction.tex').write_text(section)
    summary.update(matched64_reference_comparison_attached=True,
        matched64_primary_differences=audit['comparisons'],
        final3B_point_estimates_below_v8=top < 0 and net < 0, allocation=allocation)
    write(out/'summary-private.json', summary)
    files = {**inputs, str(binding_path):sha(binding_path), str(Path(__file__)):sha(__file__)}
    for path in out.rglob('*'):
        if path.is_file(): files[str(path)] = sha(path)
    write(out/'completion-binding-private.json', dict(files=files))
    print(dict(status='actual_matched64_reference_comparison_attached',
        candidate_identity=audit['final_identity'], new_games=0, model_loaded=False,
        student_updates=0), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); attach(args.material, args.out)
