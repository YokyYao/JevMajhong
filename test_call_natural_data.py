"""Exercise fresh natural data generation on simulator fixtures, never expert data."""
from collections import Counter
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from audit_shard import split_for_match
from call_policy import digest, request_hash


class NaturalCallDataTests(unittest.TestCase):
    def test_natural_splits_without_quotas_exclude_previous_matches(self):
        import zstandard
        root = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            groups = {}
            blobs = []
            for split, policy in [('train', 'call'), ('validation', 'pass'), ('test', 'mixed')]:
                match = next(f'natural-fixture-{split}-{i}' for i in range(10000)
                             if split_for_match(f'natural-fixture-{split}-{i}') == split)
                groups[split] = match
                destination = directory / split
                subprocess.run([sys.executable, str(root / 'desktop/audit_replay_calls.py'),
                                '--out', str(destination), '--policy', policy],
                               cwd=root, capture_output=True, text=True, check=True)
                blobs.append((match, zstandard.ZstdCompressor().compress(
                    (destination / 'private-simulated-replay.jsonl').read_bytes())))
            prior = directory / 'prior.json'
            prior.write_text(json.dumps({'matches': {'train': ['excluded-fixture']}}))
            archive_path = directory / 'fixture.tar'
            with tarfile.open(archive_path, 'w') as archive:
                for match, blob in [('excluded-fixture', blobs[0][1]), *blobs]:
                    member = tarfile.TarInfo(match + '.mjai.json.zst'); member.size = len(blob)
                    archive.addfile(member, io.BytesIO(blob))
            out = directory / 'dataset'
            config = directory / 'config.json'
            config.write_text(json.dumps({'data': str(out), 'archive': str(archive_path),
                'exclude_provenance': [str(prior)], 'training_sampling': 'natural',
                'train_records': 8, 'validation_records': 8, 'test_records': 8, 'max_per_match': 8}))
            subprocess.run([sys.executable, str(root / 'scripts/prepare_call_repair.py'),
                            '--config', str(config)], cwd=root, capture_output=True, text=True, check=True)
            audit = json.loads((out / 'provenance.json').read_text())
            self.assertTrue(audit['complete'])
            self.assertEqual(audit['counts'], {'train': 8, 'validation': 8, 'test': 8})
            self.assertEqual(audit['excluded_prior_matches'], 1)
            self.assertIn('no class quotas', audit['training_sampling'])
            seen = set()
            for split, match in groups.items():
                self.assertEqual(audit['matches'][split], [match])
                self.assertEqual(audit['files'][split], digest(out / f'{split}.jsonl'))
                rows = [json.loads(s) for s in (out / f'{split}.jsonl').read_text().splitlines()]
                counts = Counter()
                for row in rows:
                    self.assertEqual(row['_meta']['split'], split)
                    self.assertEqual(row['_meta']['group_id'], match)
                    h = request_hash(row)
                    self.assertEqual(row['_meta']['text_sha256'], h)
                    self.assertNotIn(h, seen); seen.add(h)
                    question = row['questions']['action']
                    counts[question['criteria'][question['label']]['type']] += 1
                    for action in question['criteria'].values():
                        if action['type'] in ('chi', 'pon'):
                            self.assertEqual(len(action['consumed']), 2)
                self.assertEqual(dict(counts), audit['action_types'][split])


if __name__ == '__main__': unittest.main()
