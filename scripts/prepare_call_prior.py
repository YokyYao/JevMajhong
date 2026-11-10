"""Estimate reaction priors from fresh TRAIN matches only, before class quotas."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_shard import split_for_match
from call_policy import digest, request_hash
from head_rl import write
from scripts.prepare_call_repair import convert
from replay_call_repair import UnsafeReplayCall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); cfg = json.loads(args.config.read_text())
    from riichienv import MjaiReplay
    import zstandard
    out = ROOT / cfg['prior_data']; out.mkdir(parents=True, exist_ok=False)
    excluded = set()
    for name in cfg['prior_exclude_provenance']:
        p = json.loads((ROOT / name).read_text())
        excluded.update(m for group in p['matches'].values() for m in group)
    counts, groups, seen, failures, exclusions = Counter(), set(), set(), [], Counter()
    rows = []
    with tarfile.open(ROOT / cfg['archive'], 'r|*') as archive, tempfile.TemporaryDirectory() as temporary:
        replay_path = Path(temporary) / 'match.jsonl'
        for member in archive:
            if not member.isfile() or not member.name.endswith('.zst'): continue
            match = Path(member.name).name.removesuffix('.mjai.json.zst')
            if match in excluded or split_for_match(match) != 'train': continue
            try:
                if member.size > 10_000_000: raise ValueError('Oversized compressed game')
                with archive.extractfile(member) as blob, zstandard.ZstdDecompressor().stream_reader(blob) as stream:
                    raw = stream.read(10_000_001)
                if len(raw) > 10_000_000: raise ValueError('Oversized decompressed game')
                replay_path.write_bytes(raw); replay = MjaiReplay.from_jsonl(str(replay_path), rule='mjsoul')
                pool = []
                for hand in replay.take_kyokus():
                    for player, obs, expert in hand.steps(seat=None, skip_single_action=False):
                        if json.loads(expert.to_mjai())['type'] not in ('none', 'chi', 'pon') or len(obs.hand) % 3 != 1 or obs.last_discard is None: continue
                        try: result = convert(obs, expert)
                        except UnsafeReplayCall as exc:
                            exclusions[str(exc)] += 1; continue
                        if result is None: continue
                        record, label, relative = result
                        record['questions']['action']['label'] = label
                        record['_meta'] = {'group_id': match, 'split': 'train', 'reaction_relative_seat': relative}
                        pool.append(record)
                random.Random(int(hashlib.sha256(match.encode()).hexdigest()[:16], 16)).shuffle(pool)
            except Exception as exc:
                failures.append({'match': match, 'error': str(exc)[:250]}); continue
            # The limit and permutation do not inspect expert labels.
            for row in pool[:cfg['prior_max_per_match']]:
                h = request_hash(row)
                if h in seen: continue
                seen.add(h); groups.add(match); rows.append(row)
                q = row['questions']['action']; counts[q['criteria'][q['label']]['type']] += 1
                if len(rows) == cfg['prior_records']: break
            if len(rows) == cfg['prior_records']: break
    (out / 'reactions-private.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    result = {'complete': len(rows) == cfg['prior_records'], 'n': len(rows), 'class_counts': dict(counts),
        'priors': {k: counts[k] / max(1, len(rows)) for k in ('chi', 'pon', 'none')},
        'matches': {'train': sorted(groups)}, 'excluded_previous_matches': len(excluded),
        'failed_matches': failures, 'exclusions': dict(exclusions), 'validation_or_test_matches_opened': 0,
        'sampling': 'natural offered reactions, fixed match-hash TRAIN partition; no class quotas',
        'source_archive_sha256': digest(ROOT / cfg['archive']),
        'records_sha256': digest(out / 'reactions-private.jsonl')}
    write(out / 'provenance.json', result)
    if not result['complete'] or any(counts[k] < 10 for k in ('chi', 'pon', 'none')):
        raise ValueError('Insufficient fresh training prior data')
    print(json.dumps({k: result[k] for k in ('complete', 'n', 'class_counts', 'priors')}), flush=True)


if __name__ == '__main__': main()
