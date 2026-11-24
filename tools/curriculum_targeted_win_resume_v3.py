"""Continue fixed label shards with complete-world durability and no relabeling."""
import argparse
import ast
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools import curriculum_targeted_win_resume_v2 as previous
from tools.curriculum_targeted_world_cache_v1 import CompleteWorldCache, METHOD
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.capture_curriculum_paired_terminal_v2 import resource, scheduler, terminal_record, fields
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

RUN = ROOT/'runs/curriculum-targeted-win-resume-v3'
BINDING = RUN/'entry-binding-private.json'
ENTRY = ROOT/'cluster/curriculum-targeted-win-resume-v3.slurm'


def world_loop(tree):
    _, cases = previous.original_loop(tree)
    loops = [n for n in ast.walk(cases) if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
        and [getattr(v, 'id', None) for v in n.target.elts] == ['world', 'allocation']]
    if len(loops) != 1: raise ValueError('One exact original allocation/world loop required')
    return loops[0]


def collector_tree(source):
    tree = previous.resumed_function_tree(source); loop = world_loop(tree)
    original = copy.deepcopy(loop.body)
    if (not isinstance(original[0], ast.Assign) or not isinstance(original[0].value, ast.Call)
            or not isinstance(original[0].value.func, ast.Name) or original[0].value.func.id != 'construct_world'):
        raise ValueError('Construct/verify the actual hidden world before every cache lookup')
    prefix = ast.parse("""
_resume_world_context = world_cache.make_context(spec, block_index, world, allocation, receipt,
    planned, planner_counts, seat, list(fork.scores()))
_resume_world_hit = world_cache.lookup(_resume_world_context)
if _resume_world_hit is not None:
    planner_counts.clear()
    planner_counts.update(_resume_world_hit['planner_after'])
    counts['complete_outer_arms'] += len(_resume_world_hit['paired'])
    block_pairs.append(_resume_world_hit['paired'])
    block_plans.append(dict(receipt=receipt, future_plans=_resume_world_hit['plans']))
    continue
""").body
    suffix = ast.parse("world_cache.publish(_resume_world_context, paired, plans, planner_counts)").body
    loop.body = [original[0], *prefix, *original[1:], *suffix]
    restored = [loop.body[0], *loop.body[1+len(prefix):-len(suffix)]]
    if ast.dump(ast.Module(body=restored, type_ignores=[])) != ast.dump(ast.Module(body=original, type_ignores=[])):
        raise ValueError('Original per-world model/sampling/rollout/observer calculations changed')
    return ast.fix_missing_locations(tree)


def driver_tree(source):
    tree = ast.parse(source); function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'resume')
    scope_changes = 0; output_changes = 0
    for node in function.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'scope'):
            if not isinstance(node.value, ast.Call) or any(k.arg == 'world_cache' for k in node.value.keywords):
                raise ValueError('Expected original exclusive collector scope')
            node.value.keywords.append(ast.keyword(arg='world_cache',
                value=ast.parse('make_cache(out, priors, reg)', mode='eval').body)); scope_changes += 1
    for index, node in enumerate(function.body):
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == 'write'
                and ast.unparse(node.value.args[0]) == "out / 'resumption-complete-private.json'"):
            function.body[index:index] = ast.parse("_world_manifest = scope['world_cache'].seal(out/'world-cache-binding-private.json')").body
            payload = node.value.args[1]
            if not isinstance(payload, ast.Call): raise ValueError('Expected original completion payload')
            payload.keywords.extend(ast.keyword(arg=k, value=ast.parse(v, mode='eval').body) for k, v in (
                ('world_cache_method', 'WORLD_CACHE_METHOD'), ('world_cache_manifest', 'str(_world_manifest)'),
                ('world_cache_manifest_sha256', 'sha(_world_manifest)')))
            output_changes += 1; break
    if scope_changes != 1 or output_changes != 1: raise ValueError('Expected exactly one scope and completion binding adaptation')
    return ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))


def require_terminal(folder, archive=None):
    folder = folder.resolve()
    if RUN.resolve() not in folder.parents: return previous.require_terminal(folder, archive)
    launch = read(folder/'registered-private.json'); job = str(launch['job_id'])
    receipt_path = ROOT/'logs'/f'curriculum-targeted-win-resume-v3-exit-{job}.json'; receipt = read(receipt_path)
    actual = scheduler(job)
    if actual.returncode:
        if archive is None: raise ValueError('Actual v3 prior terminal expired; never invent it')
        saved = read(archive)
        if saved['exit_receipt'] != receipt: raise ValueError('Archived v3 prior exit differs')
        raw = saved['raw']
    else: raw = actual.stdout
    proof = terminal_record(raw, job, ENTRY)
    if receipt['job_id'] != job or fields(raw).get('NodeList') != launch['node'].split('.')[0]:
        raise ValueError('Actual v3 prior job/node/exit required')
    if proof['successful']: scheduler_exit(receipt, raw, ENTRY)
    return dict(**proof, exit_receipt=receipt, receipt_path=str(receipt_path.resolve()))


def make_cache(out, priors, reg):
    from tools.curriculum_endgame_ledger import ledger
    old = [p/'world-cache-private' for p in priors if RUN.resolve() in p.resolve().parents
        and (p/'world-cache-private').is_dir()]
    identity = dict(collection_sha256=sha(previous.RUN/'registered-private.json'),
        teacher_identity=reg['continuation_model']['identity_sha256'], producer_binding_sha256=sha(BINDING),
        purpose='actual_fixed_label_collection', world_cache_method=METHOD)
    return CompleteWorldCache(out/'world-cache-private', old, identity, ledger)


def resume(shard, attempt, priors, archives):
    resource()
    binding = read(BINDING)
    if binding.get('world_cache_method') != METHOD: raise ValueError('Explicit frozen complete-world checkpointing required')
    scope = dict(vars(previous), RESUME=RUN, BINDING=BINDING, require_terminal=require_terminal,
        resumed_function_tree=collector_tree, make_cache=make_cache, WORLD_CACHE_METHOD=METHOD)
    source = Path(previous.__file__).read_text()
    exec(compile(driver_tree(source), str(previous.__file__), 'exec'), scope)
    scope['resume'](shard, attempt, [p.resolve() for p in priors], [p.resolve() if p else None for p in archives])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--shard', type=int, required=True)
    p.add_argument('--attempt', type=int, required=True); p.add_argument('--prior', type=Path, action='append', required=True)
    p.add_argument('--archive', type=Path, action='append')
    a = p.parse_args(); resume(a.shard, a.attempt, a.prior, a.archive or [None]*len(a.prior))
