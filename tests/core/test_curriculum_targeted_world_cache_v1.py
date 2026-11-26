"""Cache/software contracts; fake payload fixtures are never native labels."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools import curriculum_targeted_world_cache_v1 as cache
from tools import curriculum_targeted_win_resume_v3 as resume
from tools import curriculum_targeted_win_resume_v2 as previous


def fixture():
    identity = dict(purpose='software-fixture-only', teacher='fixed')
    planned = dict(candidates=['win', 'continue'], baseline='win', public_seed=123)
    c = cache.context(identity, dict(id='fixture', group='family', split='train'), 0, 0,
        {'hands': {1: [1, 2]}}, {'verified': True}, planned, {}, 0, [25000]*4)
    def result(scores, rank, offers):
        return dict(full_match_completed=True, opponents_no_legal_win_declines_verified=True,
            terminal_scores=scores, rank=rank, top2=rank <= 2, net_from_anchor=scores[0]-25000,
            observer_hook_calls=offers, merged_ledger_hands=1, merged_mjai_log=['fake-unit-ledger'])
    paired = dict(win=result([26000, 25000, 25000, 24000], 1, 1),
        continue_=result([24000, 26000, 25000, 25000], 4, 0))
    paired['continue'] = paired.pop('continue_')
    plans = dict(win=[dict(full_terminal_arms=8, accept='hora', decline='none', decision={'selected': 'hora'})])
    plans['continue'] = []
    after = dict(legal_win_offers=1, planned_calls=1, inner_terminal_arms=8, selected_declines=0)
    return identity, c, paired, plans, after


def validator(events, scores):
    if events != ['fake-unit-ledger']: raise ValueError('Unexpected synthetic fixture')
    return {'hands': 1}


class CompleteWorldCacheTests(unittest.TestCase):
    def test_roundtrip_preserves_all_arms_and_diagnostic_state_without_execution(self):
        identity, c, pairs, plans, after = fixture(); before = copy.deepcopy((c, pairs, plans, after))
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); writer = cache.CompleteWorldCache(root/'old', [], identity, validator)
            self.assertIsNone(writer.lookup(c)); writer.publish(c, pairs, plans, after)
            reader = cache.CompleteWorldCache(root/'new', [root/'old'], identity, validator)
            packet = reader.lookup(c)
            self.assertEqual(packet['paired'], pairs); self.assertEqual(packet['planner_after'], after)
            self.assertEqual(reader.stats['reused_worlds'], 1)
            reader.seal(root/'binding.json')
            self.assertEqual((c, pairs, plans, after), before)

    def test_partial_files_are_never_hits_and_complete_slots_are_never_overwritten(self):
        identity, c, pairs, plans, after = fixture()
        with tempfile.TemporaryDirectory() as value:
            root = Path(value); folder = root/'cache'; folder.mkdir()
            partial = folder/(cache.digest(cache.slot(c))+'.json.partial-crash'); partial.write_text('{')
            store = cache.CompleteWorldCache(folder, [], identity, validator)
            self.assertIsNone(store.lookup(c)); store.publish(c, pairs, plans, after)
            target = folder/(cache.digest(cache.slot(c))+'.json'); original = target.read_bytes()
            with self.assertRaises(ValueError): store.publish(c, pairs, plans, after)
            self.assertEqual(target.read_bytes(), original); self.assertEqual(partial.read_text(), '{')

    def test_changed_context_or_conflicting_completed_worlds_are_rejected(self):
        identity, c, pairs, plans, after = fixture()
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            first = cache.CompleteWorldCache(root/'a', [], identity, validator); first.publish(c, pairs, plans, after)
            altered = copy.deepcopy(c); altered['allocation']['hands']['1'][0] = 99
            with self.assertRaises(ValueError): first.lookup(altered)
            other = copy.deepcopy(pairs); other['win']['terminal_scores'][0] += 100
            other['win']['net_from_anchor'] += 100
            cache.CompleteWorldCache(root/'b', [], identity, validator).publish(c, other, plans, after)
            reader = cache.CompleteWorldCache(root/'c', [root/'a', root/'b'], identity, validator)
            with self.assertRaises(ValueError): reader.lookup(c)

    def test_incomplete_menu_match_or_diagnostics_cannot_be_published(self):
        identity, c, pairs, plans, after = fixture()
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            for kind in ('arm', 'match', 'observer', 'score', 'diagnostic'):
                p = copy.deepcopy(pairs); a = copy.deepcopy(after)
                if kind == 'arm': p.pop('continue')
                elif kind == 'match': p['win']['full_match_completed'] = False
                elif kind == 'observer': p['win']['observer_hook_calls'] += 1
                elif kind == 'score': p['win']['net_from_anchor'] += 1
                else: a['planned_calls'] += 1
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    cache.CompleteWorldCache(root/kind, [], identity, validator).publish(c, p, plans, a)

    def test_json_normalization_is_stable_and_rejects_nonfinite_values(self):
        a = {'players': {1: (1, 2), 3: (3, 4)}}
        b = {'players': {'1': [1, 2], '3': [3, 4]}}
        self.assertEqual(cache.digest(a), cache.digest(b))
        with self.assertRaises(ValueError): cache.digest({'bad': float('nan')})

    def test_entire_original_world_calculation_AST_is_unchanged(self):
        source = previous.COLLECTOR.read_text()
        before = resume.world_loop(previous.resumed_function_tree(source)).body
        after = resume.world_loop(resume.collector_tree(source)).body
        restored = [after[0], *after[4:-1]]
        self.assertEqual(ast.dump(ast.Module(body=before, type_ignores=[])),
                         ast.dump(ast.Module(body=restored, type_ignores=[])))

    def test_outer_resume_changes_only_cache_scope_and_manifest_attachment(self):
        source = Path(previous.__file__).read_text()
        original = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'resume')
        adapted = copy.deepcopy(resume.driver_tree(source).body[0])
        for node in adapted.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'scope':
                node.value.keywords = [k for k in node.value.keywords if k.arg != 'world_cache']
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'write':
                if ast.unparse(node.value.args[0]) == "out / 'resumption-complete-private.json'":
                    node.value.args[1].keywords = [k for k in node.value.args[1].keywords
                        if k.arg not in ('world_cache_method', 'world_cache_manifest', 'world_cache_manifest_sha256')]
        adapted.body = [n for n in adapted.body if not (isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == '_world_manifest')]
        self.assertEqual(ast.dump(original), ast.dump(adapted))


if __name__ == '__main__': unittest.main()
