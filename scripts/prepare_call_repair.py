"""Prepare fresh, answer-independent reaction supervision from expert games."""
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
from replay_call_repair import public_reaction_actions, canonical_action, canonical_signature, UnsafeReplayCall
from semantic import describe, INSTRUCTION


def convert(obs, expert):
    # Limit this pilot to replay states with an actual offered call. Rebuilding
    # calls in arbitrary singleton-pass states also needs last-draw/phase rules.
    source_kinds = {json.loads(a.to_mjai())['type'] for a in obs.legal_actions()}
    if not source_kinds & {'chi', 'pon'}:
        return None
    # Candidate generation receives only public observation, never expert.
    actions = public_reaction_actions(obs)
    if len(actions) < 2:
        return None
    state, _, d = describe(obs)
    criteria = {f'a{i}': canonical_action(a) for i, a in enumerate(actions)}
    request = {'state': state, 'questions': {'action': {
        'type': 'choice', 'instructions': INSTRUCTION, 'criteria': criteria}}}
    signature = canonical_signature(expert)
    signatures = [canonical_signature(a) for a in actions]
    if signature not in signatures:
        raise UnsafeReplayCall('expert_absent_from_independent_candidates')
    source = next(p for p, river in enumerate(d['discards']) if river and river[-1] == obs.last_discard)
    return request, f'a{signatures.index(signature)}', (source - obs.player_id) % 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    from riichienv import MjaiReplay
    import zstandard
    out = ROOT / cfg['data']
    out.mkdir(parents=True, exist_ok=False)
    excluded = set()
    for name in cfg['exclude_provenance']:
        provenance = json.loads((ROOT / name).read_text())
        excluded.update(m for group in provenance['matches'].values() for m in group)
    quotas = cfg['train_quotas']
    limits = {'train': sum(quotas.values()), 'validation': cfg['validation_records'], 'test': cfg['test_records']}
    counts = Counter()
    types = {s: Counter() for s in limits}
    groups = {s: set() for s in limits}
    exclusions, failed, hashes, selected_groups = Counter(), [], set(), Counter()
    handles = {s: (out / f'{s}.jsonl').open('w') for s in limits}
    def complete():
        return all(counts[s] == limits[s] for s in limits)
    try:
        with tarfile.open(ROOT / cfg['archive'], 'r|*') as archive, tempfile.TemporaryDirectory() as tmp:
            replay_file = Path(tmp) / 'match.jsonl'
            for member in archive:
                if not member.isfile() or not member.name.endswith('.zst'):
                    continue
                match = Path(member.name).name.removesuffix('.mjai.json.zst')
                split = split_for_match(match)
                if match in excluded or counts[split] == limits[split]:
                    continue
                if selected_groups[match]:
                    raise ValueError('Repeated source match identifier')
                try:
                    if member.size > 10_000_000:
                        raise ValueError('Oversized compressed match')
                    with archive.extractfile(member) as blob, zstandard.ZstdDecompressor().stream_reader(blob) as stream:
                        raw = stream.read(10_000_001)
                    if len(raw) > 10_000_000:
                        raise ValueError('Oversized decompressed match')
                    replay_file.write_bytes(raw)
                    replay = MjaiReplay.from_jsonl(str(replay_file), rule='mjsoul')
                    pool = []
                    for hand in replay.take_kyokus():
                        for player, obs, expert in hand.steps(seat=None, skip_single_action=False):
                            kind = json.loads(expert.to_mjai())['type']
                            if kind not in ('chi', 'pon', 'none') or len(obs.hand) % 3 != 1 or obs.last_discard is None:
                                continue
                            try:
                                result = convert(obs, expert)
                            except UnsafeReplayCall as exc:
                                exclusions[str(exc)] += 1
                                continue
                            if result is not None:
                                pool.append((*result, kind))
                    # Natural holdouts: sampling is independent of expert target.
                    random.Random(int(hashlib.sha256(match.encode()).hexdigest()[:16], 16)).shuffle(pool)
                    local_hashes, chosen = set(), []
                    local_types = Counter()
                    for request, label, relative, kind in pool:
                        if len(chosen) >= cfg['max_per_match']:
                            break
                        if split == 'train' and types[split][kind] + local_types[kind] >= quotas[kind]:
                            continue
                        h = request_hash(request)
                        if h in hashes or h in local_hashes:
                            continue
                        request['questions']['action']['label'] = label
                        request['_meta'] = {'id': f'{match}/call/{h[:20]}', 'group_id': match,
                            'split': split, 'schema': 'semantic-v3-reaction-only', 'text_sha256': h,
                            'reaction_relative_seat': relative, 'source': cfg['archive'], 'license': 'CC-BY-4.0'}
                        chosen.append((request, h, kind)); local_hashes.add(h); local_types[kind] += 1
                except Exception as exc:
                    failed.append({'match': match, 'error': str(exc)[:250]})
                    continue
                for request, h, kind in chosen:
                    if counts[split] == limits[split]:
                        break
                    handles[split].write(json.dumps(request, separators=(',', ':')) + '\n')
                    hashes.add(h); groups[split].add(match); counts[split] += 1; types[split][kind] += 1
                    selected_groups[match] += 1
                if counts[split] % 256 < cfg['max_per_match']:
                    for handle in handles.values(): handle.flush()
                    status = {'status': 'preparing_repaired_expert_reactions', 'counts': dict(counts),
                        'types': {s: dict(v) for s, v in types.items()}, 'failed_matches': len(failed)}
                    write(out / 'status.json', status); print(json.dumps(status), flush=True)
                if complete(): break
    finally:
        for handle in handles.values(): handle.close()
    audit = {'complete': complete(), 'counts': dict(counts), 'requested': limits,
        'action_types': {s: dict(v) for s, v in types.items()}, 'matches': {s: sorted(v) for s, v in groups.items()},
        'excluded_prior_matches': len(excluded), 'exclusions': dict(exclusions), 'failed_matches': failed,
        'max_decisions_per_match': max(selected_groups.values(), default=0),
        'archive_sha256': digest(ROOT / cfg['archive']),
        'code_sha256': {p: digest(ROOT / p) for p in ['scripts/prepare_call_repair.py', 'replay_call_repair.py', 'semantic.py']},
        'files': {s: digest(out / f'{s}.jsonl') for s in limits},
        'holdout_sampling': 'natural reaction distribution; match-hash splits; label-independent per-match shuffle',
        'training_sampling': 'predeclared class quotas; max decisions per match; no duplication',
        'schema': 'semantic-v3-reaction-only', 'all_kan_or_possible_ron_states_explicitly_excluded': True}
    audit['reaction_scope'] = 'source offers chi/pon; independent reconstruction never reads expert choice; singleton-pass phase legality not inferred'
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if groups[a] & groups[b]: raise ValueError('Match split overlap')
    write(out / 'provenance.json', audit)
    if not complete(): raise ValueError('Insufficient expert data; do not train on partial splits')
    write(out / 'status.json', {'status': 'prepared_and_audited', 'counts': dict(counts)})
    print(json.dumps({'status': 'prepared_and_audited', 'counts': dict(counts), 'types': audit['action_types']}), flush=True)


if __name__ == '__main__':
    main()
