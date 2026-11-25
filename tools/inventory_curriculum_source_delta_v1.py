"""Inventory new source declarations without rereading already closed payloads.

This is preparation for the final complete-source producer, never its proof.
New files still require typed stored-value, family and native-seed observations.
Run only inside a finite scheduler GPU allocation. Outputs are private.
"""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_confirmation_json_sources_v1 import _object, _constant
from tools.curriculum_transfer import sha

DIGEST = re.compile(r'[0-9a-f]{64}')
MAX_METADATA_BYTES = 2 * 1024 * 1024
BASE_STATUS = 'completed_placement_base_sources_and_families_joined_not_final_candidate'


def strict_read(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=_object, parse_constant=_constant)


def filelike(name):
    return isinstance(name, str) and ('/' in name or Path(name).suffix in (
        '.json', '.jsonl', '.py', '.toml', '.safetensors', '.pt', '.pth', '.so',
        '.slurm', '.jinja', '.md', '.log', '.txt', '.sqlite', '.zst'))


class Inventory:
    def __init__(self, known, history, root=ROOT):
        self.known = known
        self.history = history
        self.root = Path(root).resolve()
        self.nodes = {}
        self.edges = []
        self.queue = deque()
        self.parsed = set()
        self.issues = []
        self.hashed = {}

    def add(self, name, expected, owner, pointer, base=None):
        if not isinstance(expected, str) or DIGEST.fullmatch(expected) is None:
            return
        base = self.root if base is None else Path(base)
        path = Path(name)
        if not path.is_absolute():
            path = base / path
            if not path.is_file() and base == self.root:
                local = Path(owner).parent / name
                if local.is_file():
                    path = local
        path = path.resolve()
        alias = None
        h = self.history
        if (owner == h['original_manifest'] and str(path) == h['original_declaration_path']
                and expected == h['recovered_original_sha256']):
            alias = str(path)
            path = Path(h['recovered_original_path']).resolve()
        key = str(path)
        issue = dict(owner=owner, pointer=pointer, path=key, expected=expected)
        if not path.is_file():
            self.issues.append(dict(issue, issue='missing_declared_file'))
            return
        if key in self.nodes and self.nodes[key]['sha256'] != expected:
            self.issues.append(dict(issue, issue='conflicting_declared_identity',
                previous=self.nodes[key]['sha256']))
            return
        reused = self.known.get(key) == expected
        before = path.stat()
        hashed = False
        if not reused and before.st_size <= MAX_METADATA_BYTES:
            if key not in self.hashed:
                self.hashed[key] = sha(path)
            if self.hashed[key] != expected:
                self.issues.append(dict(issue, issue='declared_file_hash_mismatch',
                    actual=self.hashed[key]))
                return
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('A source changed while its small-file identity was observed')
            hashed = True
        if key not in self.nodes:
            self.nodes[key] = dict(path=key, sha256=expected, suffix=path.suffix,
                size_bytes=before.st_size, reused_completed_source_observation=reused,
                actual_bytes_hashed_in_this_inventory=hashed,
                typed_source_observation_pending=not reused)
            if not reused and hashed and path.suffix == '.json':
                self.queue.append(path)
        edge = dict(owner=owner, pointer=pointer, path=key, sha256=expected)
        if alias is not None:
            edge['exact_previously_proved_historical_declaration'] = alias
        self.edges.append(edge)

    def walk(self, value, owner, pointer='', base=None):
        base = self.root if base is None else Path(base)
        if isinstance(value, list):
            for index, child in enumerate(value):
                self.walk(child, owner, pointer + '/' + str(index), base)
        elif isinstance(value, dict):
            localbase = self.root / value['directory'] if isinstance(value.get('directory'), str) else base
            for field in ('path', 'file', 'training_file', 'validation_file', 'protocol'):
                name = value.get(field)
                digest = value.get(field + '_sha256',
                    value.get('sha256') if field in ('path', 'file') else None)
                if isinstance(name, str):
                    self.add(name, digest, owner, pointer + '/' + field)
            for key, child in value.items():
                where = pointer + '/' + str(key).replace('~', '~0').replace('/', '~1')
                if filelike(key) and isinstance(child, str) and DIGEST.fullmatch(child):
                    self.add(key, child, owner, where, base)
                else:
                    self.walk(child, owner, where, localbase if key == 'files' else self.root)

    def expand(self):
        while self.queue:
            path = self.queue.popleft()
            key = str(path)
            if key in self.parsed:
                continue
            # The root hash is checked again immediately before parsing. Large
            # newly declared JSON remains explicitly pending for a later pass.
            if sha(path) != self.nodes[key]['sha256']:
                raise ValueError('New source metadata changed before declaration traversal')
            self.parsed.add(key)
            self.walk(strict_read(path), key)
            if len(self.nodes) > 20000 or len(self.parsed) > 10000:
                raise ValueError('Review an unexpectedly expansive source graph')


def inventory(base, roots, out):
    allocation = resource()
    started = time.monotonic()
    base, out = Path(base).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Never overwrite a complete or partial source inventory')
    binding_path = base / 'completion-binding-private.json'
    binding = strict_read(binding_path)['files']
    files = {str(binding_path): sha(binding_path), str(Path(__file__).resolve()): sha(__file__)}

    def checked(path):
        path = Path(path).resolve()
        actual = sha(path)
        if binding.get(str(path)) != actual:
            raise ValueError('Actual completed base evidence changed: ' + str(path))
        files[str(path)] = actual
        return strict_read(path)

    summary = checked(base / 'summary-private.json')
    if (summary['status'] != BASE_STATUS or summary['reserved_source_data_hits']
            or summary['unreviewed_artifacts']):
        raise ValueError('Actual completed full base-source evidence required')
    known = checked(base / 'actual-artifact-files-private.json')
    if len(known) != summary['actual_conservative_ancestral_and_prepared_base_artifacts']:
        raise ValueError('Completed source catalogue count differs')
    history = checked(ROOT / 'runs/curriculum-active-student-generic-config-history-v3/summary-private.json')
    graph = Inventory(known, history)
    roots = sorted({str(Path(path).resolve()) for path in roots})
    if not roots:
        raise ValueError('Explicit actual completed metadata roots required')
    for name in roots:
        path = Path(name)
        if path.suffix != '.json' or path.stat().st_size > MAX_METADATA_BYTES:
            raise ValueError('Root must be explicit bounded JSON metadata; no directory guessing')
        actual = sha(path)
        files[name] = actual
        graph.add(name, actual, name, '/explicit_actual_root')
    graph.expand()
    additional = {name: row for name, row in graph.nodes.items()
        if not row['reused_completed_source_observation']}
    unread = {name: row for name, row in additional.items()
        if not row['actual_bytes_hashed_in_this_inventory']}
    result = dict(status='actual_source_delta_inventoried_not_complete_source_proof',
        explicit_root_files=roots, reused_base_catalogue_assets=len(known),
        actual_declared_artifacts=len(graph.nodes), metadata_hash_edges=len(graph.edges),
        parsed_new_JSON_metadata=len(graph.parsed), additional_artifacts=len(additional),
        additional_suffix_counts=dict(Counter(row['suffix'] for row in additional.values())),
        unread_large_artifacts=len(unread), declaration_issues=len(graph.issues),
        old_closed_source_payloads_not_rescanned=True, typed_new_source_observations_still_required=True,
        full_source_closure=False, final_confirmation_admitted=False,
        reserved_games_generated=0, new_games=0, model_loaded=False, optimizer_updates=0,
        allocation=allocation, seconds=time.monotonic()-started)
    out.mkdir(parents=True, exist_ok=False)
    for name, value in (
            ('declared-artifact-nodes-private.json', graph.nodes),
            ('metadata-hash-edges-private.json', graph.edges),
            ('additional-artifacts-pending-observation-private.json', additional),
            ('unread-large-artifacts-private.json', unread),
            ('declaration-issues-private.json', graph.issues),
            ('summary-private.json', result)):
        write(out / name, value)
        files[str(out / name)] = sha(out / name)
    files.update(graph.hashed)
    write(out / 'completion-binding-private.json', dict(files=files,
        actual_final_candidate_admitted=False, unread_payloads_not_claimed_verified=True))
    print(json.dumps({key: value for key, value in result.items()
        if key not in ('allocation', 'explicit_root_files')}), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-closure', type=Path, required=True)
    p.add_argument('--root', type=Path, action='append', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    inventory(args.base_closure, args.root, args.out)
