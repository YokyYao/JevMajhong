"""Resume fixed targeted labels after a genuine terminal; preserve old artifacts.

The original bound case loop, teacher, worlds and targets are executed unchanged.
Only the output directory and a completed-case skip are added. Partial cases
are retained as crash artifacts and never admitted as labels.
"""
import argparse
import ast
from collections import Counter
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, fields, terminal_record

RUN = ROOT/'runs/curriculum-targeted-win-labels-v1'
RESUME = ROOT/'runs/curriculum-targeted-win-resume-v2'
COLLECTOR = ROOT/'tools/collect_curriculum_targeted_win_labels_v1.py'
ENTRY = ROOT/'cluster/curriculum-targeted-win-resume-v2.slurm'
BINDING = RESUME/'entry-binding-private.json'


def original_loop(tree):
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'collect')
    loops = [n for n in ast.walk(function) if isinstance(n, ast.For)
        and isinstance(n.target, ast.Tuple)
        and [getattr(e, 'id', None) for e in n.target.elts] == ['index', 'spec']
        and ast.dump(n.iter) == ast.dump(ast.parse('enumerate(specs)', mode='eval').body)]
    if len(loops) != 1: raise ValueError('One exact original fixed-view case loop required')
    return function, loops[0]


def resumed_function_tree(source):
    tree = ast.parse(source); function, loop = original_loop(tree)
    original_body = ast.dump(ast.Module(body=copy.deepcopy(loop.body), type_ignores=[]))
    replacements = 0
    for node in ast.walk(function):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'out'):
            if ast.unparse(node.value) != "root / 'shards' / f'shard-{shard:02d}'":
                raise ValueError('Original output expression changed')
            node.value = ast.Name(id='RESUME_OUT', ctx=ast.Load()); replacements += 1
    if replacements != 1: raise ValueError('Only the original output directory may change')
    skip = ast.parse("""
if spec['id'] in RECOVERED:
    reuse(RECOVERED[spec['id']], spec, index, case_dir, labels, quarantine, counts, contexts)
    print(json.dumps(dict(stage='targeted_win_reused_complete_case', shard=shard, index=index,
        source_id=spec['id'])), flush=True)
    continue
""").body
    loop.body[:0] = skip
    if ast.dump(ast.Module(body=loop.body[len(skip):], type_ignores=[])) != original_body:
        raise ValueError('Current/future rollout, sampling, scoring or target logic changed')
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def audit_case(case, spec, reg):
    from tools.curriculum_rank_win_controls_v1 import soft_target
    from tools.curriculum_endgame_ledger import ledger
    from tools.audit_curriculum_history_worlds import derived_seed
    c = reg['config']; blocks = case['blocks']
    if (case['source_id'] != spec['id'] or case['group'] != spec['group']
            or case['split'] != spec['split'] or len(blocks) != 2
            or case['teacher_model'] != reg['continuation_model']['identity_sha256']):
        raise ValueError('Recovered complete case source/teacher differs')
    first = blocks[0]
    if len(first['candidates']) != spec['legal_menu_size']:
        raise ValueError('Recovered current legal-menu size differs')
    for block in blocks:
        if (len(block['pairs']) != 16 or not block['complete_current_legal_menu']
                or block['candidates'] != first['candidates'] or block['baseline'] != first['baseline']
                or block['tested_menu_size'] != spec['legal_menu_size']):
            raise ValueError('Both complete independent16-world blocks required')
        for pair in block['pairs']:
            if set(pair) != set(first['candidates']): raise ValueError('A recovered current arm is missing')
            for outcome in pair.values():
                ledger(outcome['merged_mjai_log'], outcome['terminal_scores'])
                if not outcome['full_match_completed'] or not outcome['opponents_no_legal_win_declines_verified']:
                    raise ValueError('Incomplete observer-aware terminal outcome')
    pairs = [p for b in blocks for p in b['pairs']]
    target = soft_target(pairs, first['candidates'], spec['public_rank'],
        seed=derived_seed(first['public_seed'], 512))
    row = case['candidate_label']; meta = row['_meta']; q = row['questions']['action']
    if (target != case['soft_label'] or meta['id'] != spec['id'] or meta['group_id'] != spec['group']
            or meta['split'] != spec['split'] or meta['score_context'] != spec['context']
            or q['src'] != 'curriculum4:rank_win:targeted:'+spec['bucket']
            or q['label'] != target['label']
            or q['target'] != {k: target['target'].get(k, 0.) for k in q['criteria']}):
        raise ValueError('Recovered label does not reconstruct from its full worlds')
    return row


def completed_from(folder, specs, reg):
    """A live folder may only be inspected by preflight, never resumed directly."""
    launch = read(folder/'registered-private.json')
    if (launch['parent_registration_sha256'] != sha(RUN/'registered-private.json')
            or launch['selected'] != specs or launch['model'] != reg['continuation_model']):
        raise ValueError('Prior exact selected population and teacher required')
    recovered = {}; inputs = {str((folder/'registered-private.json').resolve()): sha(folder/'registered-private.json')}
    labels = {}
    label_path = folder/'labels-private.jsonl'
    if label_path.exists():
        with label_path.open() as stream:
            for line in stream:
                if not line.endswith('\n'): break
                row = json.loads(line); rid = row['_meta']['id']
                if rid in labels: raise ValueError('Duplicate prior emitted label')
                labels[rid] = row
    for index, spec in enumerate(specs):
        path = folder/'cases-private'/f'case-{index:03d}.json'
        if not path.exists():
            if spec['id'] in labels: raise ValueError('Emitted prior label has no full case')
            continue
        try: case = read(path)
        except json.JSONDecodeError:
            if spec['id'] in labels: raise ValueError('Emitted prior label has a truncated case')
            continue
        row = audit_case(case, spec, reg)
        if spec['id'] in labels and labels[spec['id']] != row:
            raise ValueError('Emitted prior label differs from full case')
        recovered[spec['id']] = dict(kind='case', path=str(path.resolve()), sha256=sha(path), case=case)
        inputs[str(path.resolve())] = sha(path)
    if set(labels)-set(recovered): raise ValueError('Unknown or incomplete prior emitted labels')
    path = folder/'quarantine-private.jsonl'
    if path.exists():
        expected = {s['id']: s for s in specs}
        with path.open() as stream:
            for line in stream:
                if not line.endswith('\n'): break
                q = json.loads(line); rid = q['source_id']; spec = expected[rid]
                if (rid in recovered or q['split'] != spec['split'] or q['group'] != spec['group']
                        or q['reason'] != 'public_prior_sampling_exhausted' or not q['no_partial_target_admitted']):
                    raise ValueError('Original explicit sampler quarantine must be preserved')
                recovered[rid] = dict(kind='quarantine', row=q)
    for path in (label_path, folder/'quarantine-private.jsonl'):
        if path.exists(): inputs[str(path.resolve())] = sha(path)
    return recovered, inputs


def combine(recovered, incoming):
    for rid, value in incoming.items():
        if rid in recovered:
            old = recovered[rid]
            if old['kind'] != value['kind'] or (old['case'] != value['case'] if old['kind'] == 'case' else old['row'] != value['row']):
                raise ValueError('Never choose between conflicting completed attempts')
        else: recovered[rid] = value


def reuse(value, spec, index, case_dir, labels, quarantine, counts, contexts):
    if value['kind'] == 'quarantine':
        quarantine.write(json.dumps(value['row'])+'\n'); quarantine.flush()
        counts['sampling_quarantine'] += 1
        return
    case = value['case']; path = Path(value['path'])
    if sha(path) != value['sha256']: raise ValueError('Prior complete case changed')
    (case_dir/f'case-{index:03d}.json').symlink_to(path)
    labels.write(json.dumps(case['candidate_label'], separators=(',', ':'))+'\n'); labels.flush()
    result = case['soft_label']; baseline = case['blocks'][0]['baseline']
    counts['complete_outer_arms'] += sum(len(p) for b in case['blocks'] for p in b['pairs'])
    counts['labelled_views'] += 1; counts['alternative_soft_argmax'] += result['label'] != baseline
    counts['block_agreement'] += result['independent_block_choices_agree']
    counts['leave_one_agreement'] += result['every_leave_one_world_choice_agrees']
    counts.update({'future:'+k: n for k, n in case['planner_counts'].items()})
    cc = contexts.setdefault(spec['split']+':'+spec['bucket'], Counter())
    cc['views'] += 1; cc['alternative_soft_argmax'] += result['label'] != baseline
    cc['point_alternative'] += result['point_choice'] != baseline


def require_terminal(folder, archive=None):
    launch = read(folder/'registered-private.json'); job = str(launch['job_id'])
    original = folder.parent == RUN/'shards'
    entry = ROOT/'cluster/curriculum-targeted-win-labels-v1.slurm' if original else ENTRY
    receipt_path = ROOT/'logs'/f"{'curriculum-targeted-win-labels' if original else 'curriculum-targeted-win-resume'}-exit-{job}.json"
    receipt = read(receipt_path)
    actual = scheduler(job)
    if actual.returncode:
        if archive is None: raise ValueError('Actual prior terminal expired; provide its preserved real archive')
        saved = read(archive)
        if saved['exit_receipt'] != receipt: raise ValueError('Archived prior phase exit differs')
        raw = saved['raw']
    else: raw = actual.stdout
    proof = terminal_record(raw, job, entry)
    if (receipt['job_id'] != job or (original and receipt.get('phase') != 'label')
            or fields(raw).get('NodeList') != launch['node'].split('.')[0]):
        raise ValueError('Actual original terminal/receipt/node required; never resume a live writer')
    if proof['successful']:
        from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
        scheduler_exit(receipt, raw, entry)
    return dict(**proof, exit_receipt=receipt, receipt_path=str(receipt_path.resolve()))


def resume(shard, attempt, priors, archives):
    allocation = resource(); binding = read(BINDING)
    if (binding['status'] != 'targeted_win_resume_entry_bound_not_started'
            or any(sha(p) != h for p, h in binding['files'].items())
            or shard not in range(4) or attempt < 1 or len(priors) != len(archives)):
        raise ValueError('Frozen resume entry, valid shard/attempt and prior proofs required')
    reg = read(RUN/'registered-private.json'); specs = reg['selected'][shard*27:(shard+1)*27]
    source = read(reg['source_registration'])
    sys.path[:0] = [source['source'], source['training_config']['upstream_source']]
    recovered = {}; inputs = {}; terminals = []
    original_folder = RUN/'shards'/f'shard-{shard:02d}'
    if not priors or priors[0].resolve() != original_folder.resolve():
        raise ValueError('Resume must preserve the original shard first')
    if len({p.resolve() for p in priors}) != len(priors):
        raise ValueError('Duplicate prior attempts are not a recovery lineage')
    for folder, archive in zip(priors, archives):
        terminals.append(require_terminal(folder, archive))
        incoming, bound = completed_from(folder, specs, reg); combine(recovered, incoming); inputs.update(bound)
    out = RESUME/f'shard-{shard:02d}'/f'attempt-{attempt:03d}'
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = out.parent/f'attempt-{attempt:03d}-admission-private.json'
    write(manifest, dict(status='actual_terminal_complete_cases_preserved_missing_cases_only',
        priors=[str(p.resolve()) for p in priors], terminals=terminals, files=inputs,
        completed_views=len(recovered), remaining_views=27-len(recovered), allocation=allocation,
        same_fixed_population_worlds_teacher_and_targets=True, student_updates=0))
    from tools import collect_curriculum_targeted_win_labels_v1 as original
    scope = dict(vars(original), RESUME_OUT=out, RECOVERED=recovered, reuse=reuse)
    code = compile(resumed_function_tree(COLLECTOR.read_text()), str(COLLECTOR), 'exec')
    exec(code, scope); scope['collect'](shard)
    if any(sha(p) != h for p, h in inputs.items()): raise ValueError('A reused complete case changed')
    write(out/'resumption-complete-private.json', dict(admission=str(manifest), binding_sha256=sha(BINDING),
        original_case_loop_unchanged=True, exact_recovered_views=len(recovered),
        newly_processed_views=27-len(recovered), partial_worlds_admitted=False, student_updates=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard', type=int, required=True); p.add_argument('--attempt', type=int, required=True)
    p.add_argument('--prior', type=Path, action='append', required=True)
    p.add_argument('--archive', type=Path, action='append')
    a = p.parse_args()
    resume(a.shard, a.attempt, a.prior, a.archive or [None]*len(a.prior))
