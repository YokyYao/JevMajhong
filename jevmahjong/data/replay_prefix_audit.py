"""Independently enumerate raw replay prefixes before admitting expert kan data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile

from riichienv import GameRule, RiichiEnv
import zstandard

from jevmahjong.core.io import digest, write
from jevmahjong.core.state import build_request

SCHEMA = 'public-prefix-kan-v1'
KINDS = {'ankan', 'daiminkan', 'kakan'}


def observed_action(event):
    """Strict raw MJAI signature; never repair an answer-dependent menu."""
    typ = event['type']; expected = {'chi': 2, 'pon': 2, 'daiminkan': 3, 'ankan': 4, 'kakan': 3}
    tiles = event.get('consumed', [])
    if typ in expected and len(tiles) != expected[typ]:
        raise ValueError('Raw observed meld has invalid consumed count')
    return (typ, None if typ in ('ankan', 'none', 'reach') else event.get('pai'), tuple(sorted(tiles)))


def public_input(record):
    q = record['questions']['action']
    if set(record) != {'state', 'questions', '_meta'} or set(record['questions']) != {'action'}:
        raise ValueError('Unexpected record payload')
    if set(q) != {'type', 'instructions', 'criteria', 'label'}:
        raise ValueError('Unexpected expert question annotations')
    return dict(state=record['state'], questions={'action': {k: v for k, v in q.items() if k != 'label'}})


def independent_match(events, match, member, raw_sha, rows, seed):
    """Enumerate every observed window without calling producer reconstruction."""
    starts = [e for e in events if e['type'] == 'start_kyoku']
    if not starts or any(len(e['tehais']) != 4 for e in starts) or events[-1]['type'] != 'end_game':
        raise ValueError('Complete four-player raw replay required')
    indexed = {}
    for row in rows:
        meta = row['_meta']; key = (meta['prefix_events'], row['state']['seat'])
        if key in indexed:
            raise ValueError('Duplicate prefix observer record')
        indexed[key] = row
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=seed)
    env.reset(); prior_kind = None; prior_actor = None; prefix = hashlib.sha256()
    checked = 0; kinds = Counter(); seen = set(); positions = defaultdict(set)
    for index, event in enumerate(events):
        typ = event['type']
        if typ == 'tsumo' and prior_kind == 'dahai':
            observers = [(p, {'type': 'none'}) for p in range(4) if p != prior_actor]
        elif typ in ('dahai', 'reach', 'ankan', 'kakan', 'chi', 'pon', 'daiminkan'):
            observers = [(event['actor'], event)]
        else:
            observers = []
        for player, actual in observers:
            obs = env.get_observations([player])[player]
            actions = obs.legal_actions()
            available = {json.loads(a.to_mjai())['type'] for a in actions}
            if not available & KINDS or len(actions) < 2:
                continue
            key = (index, player)
            if key not in indexed:
                raise ValueError('Missing observed prefix record')
            row = indexed[key]; meta = row['_meta']
            request, _, _ = build_request(obs)
            if public_input(row) != request:
                raise ValueError('Raw prefix public state or complete menu differs')
            q = row['questions']['action']; chosen = q['criteria'][q['label']]
            signature = observed_action(actual)
            if observed_action(chosen) != signature:
                raise ValueError('Observed replay choice differs')
            expected_meta = dict(id=f'{match}/kan/{checked}', group_id=match,
                source='raw_mjai_prefix', split='train', schema=SCHEMA,
                prefix_events=index, prefix_sha256=prefix.hexdigest(), archive_member=member,
                raw_match_sha256=raw_sha)
            if meta != expected_meta:
                raise ValueError('Raw prefix event identity differs')
            fields = ('seat', 'hand', 'rivers', 'melds', 'dora_indicators', 'discards_count', 'honba', 'sticks')
            position = hashlib.sha256(json.dumps({k: row['state'].get(k) for k in fields},
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            positions[position].add(signature)
            checked += 1; kinds[actual['type']] += 1; seen.add(key)
        env.apply_event(event)
        prefix.update(json.dumps(event, sort_keys=True, separators=(',', ':')).encode() + b'\n')
        if typ not in ('reach_accepted', 'dora'):
            prior_kind, prior_actor = typ, event.get('actor')
    if seen != set(indexed):
        raise ValueError('Extra or censored prefix record')
    return dict(records=checked, action_types=dict(kinds), unique_positions=len(positions),
        conflicting_position_labels=sum(len(v) > 1 for v in positions.values()))


def audit(cfg):
    from jevmahjong.training.base_lora import package_digest
    read = lambda p: json.loads(Path(p).read_text())
    producer = Path(cfg['producer_out']); provenance_path = producer / 'provenance.json'
    result_path = producer / 'result.json'; train = producer / 'train.jsonl'
    bound = {str(provenance_path): cfg['producer_provenance_sha256'],
        str(result_path): cfg['producer_result_sha256'], str(train): cfg['producer_train_sha256'],
        cfg['producer_launch']: cfg['producer_launch_sha256'], cfg['native_probe']: cfg['native_probe_sha256']}
    def verify():
        for name, checksum in bound.items():
            if digest(Path(name)) != checksum:
                raise ValueError('Bound source input changed: ' + name)
    verify(); provenance = read(provenance_path); result = read(result_path); launch = read(cfg['producer_launch'])
    if (provenance.get('schema') != SCHEMA or provenance.get('training_admitted') is not False
            or not provenance.get('independent_source_and_prefix_audit_required')
            or provenance.get('test_payloads_opened') is not False
            or result['status'] != 'expert_kan_prefix_extracted_pending_independent_audit'
            or result['failed_matches'] or provenance['failures']
            or result['provenance_sha256'] != cfg['producer_provenance_sha256']
            or provenance['files']['train'] != cfg['producer_train_sha256']
            or launch['command'] != 'prepare-prefix-kan' or launch['config'] != provenance['config']):
        raise ValueError('Complete unadmitted producer receipts required')
    source = Path(launch['source']) / 'jevmahjong'
    if package_digest(source) != launch['source_sha256']:
        raise ValueError('Frozen producer source changed')
    package = Path(__file__).resolve().parents[1]
    for name in ('data/replay_prefix.py', 'data/replay_calls.py', 'core/state.py', 'core/io.py'):
        if digest(source / name) != digest(package / name):
            raise ValueError('Audited producer or public format differs: ' + name)
        bound[str(package / name)] = digest(package / name)
    bound[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    qa = read(cfg['native_probe'])
    if (qa['mismatches'] or not qa['checks_shared_public_request_and_features']
            or not qa['uses_different_shadow_random_seed']
            or any(qa['action_opportunities'].get(k, 0) < 1 for k in KINDS)):
        raise ValueError('Full public native parity across three kan kinds required')
    pcfg = provenance['config']; catalog = Path(pcfg['catalog_index'])
    bound[str(catalog)] = pcfg['catalog_index_sha256']; bound.update(pcfg['archives']); verify()
    db = sqlite3.connect('file:' + str(catalog.resolve()) + '?mode=ro', uri=True)
    held = {g for (g,) in db.execute("SELECT DISTINCT group_key FROM groups WHERE split!='train'")}
    allowed = {g for (g,) in db.execute("SELECT DISTINCT group_key FROM groups WHERE split='train'")}
    allowed = {g for g in allowed - held if g.startswith('replay:')}; db.close()
    by_match = defaultdict(list); ids = set()
    with train.open() as stream:
        for line in stream:
            row = json.loads(line); meta = row['_meta']
            if 'replay:' + str(meta['group_id']) not in allowed or meta['id'] in ids:
                raise ValueError('Foreign/held group or duplicate expert record')
            ids.add(meta['id']); by_match[meta['group_id']].append(row)
    counts = Counter(); kinds = Counter(); visited = set(); nonempty = set()
    for file in pcfg['archives']:
        with tarfile.open(file, 'r|*') as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith('.mjai.json.zst'):
                    continue
                match = Path(member.name).name.removesuffix('.mjai.json.zst')
                if 'replay:' + match not in allowed or match in visited:
                    continue
                visited.add(match)
                if member.size > 10_000_000:
                    raise ValueError('Oversized raw source')
                with archive.extractfile(member) as compressed:
                    with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                        raw = reader.read(10_000_001)
                if len(raw) > 10_000_000:
                    raise ValueError('Oversized raw replay')
                events = [json.loads(line) for line in raw.splitlines() if line.strip()]
                actual = independent_match(events, match, member.name, hashlib.sha256(raw).hexdigest(),
                    by_match.get(match, []), pcfg['shadow_seed'] + 470003)
                counts['matches_replayed'] += 1
                for key in ('records', 'unique_positions', 'conflicting_position_labels'):
                    counts[key] += actual[key]
                kinds.update(actual['action_types'])
                if actual['records']:
                    nonempty.add(match)
    if (set(by_match) != nonempty or nonempty != set(provenance['matches']['train'])
            or provenance['matches']['validation'] or provenance['matches']['test']
            or counts['records'] != provenance['counts']['train']
            or counts['records'] != result['counts']['records']
            or counts['matches_replayed'] != result['counts']['matches_replayed']
            or dict(kinds) != provenance['action_types'] or dict(kinds) != result['action_types']):
        raise ValueError('Complete raw coverage, grouping or label counts differ')
    verify()
    if package_digest(source) != launch['source_sha256']:
        raise ValueError('Producer source changed during audit')
    out = Path(cfg['out']); out.mkdir(parents=True, exist_ok=False)
    write(out / 'protocol.json', dict(config=cfg, source_files=bound,
        producer_source_sha256=launch['source_sha256'], producer_source=str(source)))
    proof = dict(status='passed_complete', audit_kind='observed_expert_kan_prefix_v1',
        counts=dict(counts), action_types=dict(kinds), train_groups=len(nonempty),
        source_train_sha256=cfg['producer_train_sha256'],
        source_provenance_sha256=cfg['producer_provenance_sha256'],
        all_observed_eligible_windows_enumerated=True, inputs_rebuilt_before_label=True,
        alternate_shadow_seed=True, censored_passes_excluded=True, historical_held_groups_excluded=True,
        sealed_test_payloads_opened=False, eligible_for_pool_integration=True,
        training_started=False, expert_optimality_proven=False, new_strength_matches=0)
    write(out / 'result.json', proof); write(out / 'status.json', proof)
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    print(json.dumps(audit(json.loads(parser.parse_args().config.read_text()))))


if __name__ == '__main__':
    main()
