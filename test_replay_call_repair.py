import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from replay_call_repair import canonical_action, UnsafeReplayCall


class ReplayCallRepairTests(unittest.TestCase):
    def test_prepare_pipeline_uses_repaired_schema_and_counts_only_emitted_rows(self):
        import io
        import tarfile
        import zstandard
        from audit_shard import split_for_match
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            subprocess.run([sys.executable, str(root / 'desktop/audit_replay_calls.py'), '--out', str(directory / 'replay'),
                            '--policy', 'mixed'], cwd=root, check=True, capture_output=True)
            raw = (directory / 'replay/private-simulated-replay.jsonl').read_bytes()
            data = zstandard.ZstdCompressor().compress(raw)
            i = 0
            while split_for_match(f'call-repair-fixture-{i}') != 'train':
                i += 1
            with tarfile.open(directory / 'fixture.tar', 'w') as archive:
                member = tarfile.TarInfo(f'call-repair-fixture-{i}.mjai.json.zst')
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
            subprocess.run([sys.executable, str(root / 'prepare_raw.py'), str(directory / 'fixture.tar'),
                            '--out', str(directory / 'dataset'), '--train', '16', '--validation', '0',
                            '--test', '0', '--per-match', '128'], cwd=root, check=True, capture_output=True)
            rows = [json.loads(line) for line in (directory / 'dataset/train.jsonl').read_text().splitlines()]
            provenance = json.loads((directory / 'dataset/provenance.json').read_text())
            self.assertEqual(len(rows), 16)
            self.assertEqual(provenance['schema'], 'semantic-v2-call-repaired')
            emitted = sum(any(a['type'] in ('chi', 'pon') for a in row['questions']['action']['criteria'].values()) for row in rows)
            self.assertEqual(provenance['call_repair_audit'].get('reaction_records_rebuilt', 0), emitted)
            self.assertTrue(provenance['call_exclusions'])
            for row in rows:
                question = row['questions']['action']
                self.assertIn(question['label'], question['criteria'])
                self.assertNotIn('label', row['state'])
                for action in question['criteria'].values():
                    if action['type'] in ('chi', 'pon'):
                        self.assertEqual(len(action['consumed']), 2)

    def test_called_tile_is_removed_once_without_losing_owned_red(self):
        action = {'type': 'pon', 'pai': '5p', 'consumed': ['5pr', '5p', '5p'], 'actor': 2}
        repaired = canonical_action(action)
        self.assertEqual(repaired['consumed'], ['5p', '5pr'])
        self.assertEqual(len(action['consumed']), 3)
        self.assertNotIn('actor', repaired)
        self.assertEqual(canonical_action(repaired), repaired)

    def test_malformed_meld_fails_instead_of_inventing_tiles(self):
        with self.assertRaises(UnsafeReplayCall):
            canonical_action({'type': 'chi', 'pai': '3m', 'consumed': ['1m', '2m', '4m']})
        with self.assertRaises(UnsafeReplayCall):
            canonical_action({'type': 'pon', 'pai': 'P', 'consumed': ['P']})

    def test_repaired_replay_candidates_match_live_without_using_expert_choice(self):
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(root / 'desktop/audit_replay_calls.py'),
                                     '--out', directory, '--policy', 'mixed'],
                                    cwd=root, capture_output=True, text=True, check=True)
            summary = json.loads(result.stdout)
            comparison = summary['repaired_comparison']
            self.assertGreater(comparison['matched'], 50)
            self.assertEqual(comparison['mismatches'], 0)
            self.assertEqual(comparison.get('unmatched', 0), 0)
            self.assertGreater(comparison['alternatives_restored'], 0)
            self.assertGreater(sum(summary['explicit_exclusions'].values()), 0)
            rows = json.loads((Path(directory) / 'visible-call-pairs.json').read_text())['replay']
            targets = set()
            for row in rows:
                if 'repaired_criteria' not in row:
                    continue
                targets.add(row['chosen']['type'])
                for action in row['repaired_criteria'].values():
                    if action['type'] in ('chi', 'pon'):
                        self.assertEqual(len(action['consumed']), 2)
            self.assertTrue({'chi', 'pon', 'none'} <= targets)


if __name__ == '__main__':
    unittest.main()
