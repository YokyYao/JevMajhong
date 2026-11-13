"""Extract observed kan decisions from label-independent MJAI prefixes.

This producer does not admit its output to a training pool. Raw replay source,
public-state parity and historical splits still require an independent audit.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile

from riichienv import GameRule, RiichiEnv
import zstandard

from jevmahjong.core.io import digest, write
from jevmahjong.core.state import build_request
from jevmahjong.data.replay_calls import canonical_action

SCHEMA = 'public-prefix-kan-v1'
KAN = {'ankan', 'daiminkan', 'kakan'}
CHOICES = {'dahai', 'reach', 'ankan', 'kakan', 'chi', 'pon', 'daiminkan'}
META = {'reach_accepted', 'dora'}


def action_key(value):
    """Ignore event bookkeeping and the redundant ankan pai representation."""
    action = canonical_action(value)
    action = {k: v for k, v in action.items() if k in ('type', 'pai', 'consumed')}
    if action['type'] == 'ankan':
        action.pop('pai', None)
    return json.dumps(action, sort_keys=True)


class PrefixReplay:
    """Build each input before applying the event used as its label."""
    def __init__(self, seed=0):
        self.env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=seed)
        self.env.reset()
        self.pending_discard = None
        self.prefix = hashlib.sha256()
        self.event_count = 0

    def request(self, player):
        obs = self.env.get_observations([player])[player]
        actions = obs.legal_actions()
        kinds = {json.loads(a.to_mjai())['type'] for a in actions}
        if not kinds & KAN or len(actions) < 2:
            return None
        return build_request(obs)[0]

    def labelled(self, player, event):
        request = self.request(player)
        if request is None:
            return None
        criteria = request['questions']['action']['criteria']
        key = action_key(event)
        matches = [name for name, action in criteria.items() if action_key(action) == key]
        if len(matches) != 1:
            raise ValueError('Observed action absent or ambiguous in prefix-only kan menu')
        request['questions']['action']['label'] = matches[0]
        request['_prefix'] = dict(event_count=self.event_count, sha256=self.prefix.hexdigest())
        return request

    def consume(self, event):
        typ = event['type']
        samples = []
        if typ == 'tsumo' and self.pending_discard is not None:
            # A following draw identifies all reaction passes. Other claims
            # censor the remaining players' unrecorded decisions.
            for player in range(4):
                if player != self.pending_discard:
                    request = self.labelled(player, {'type': 'none'})
                    if request is not None:
                        samples.append(request)
        elif typ in CHOICES:
            request = self.labelled(event['actor'], event)
            if request is not None:
                samples.append(request)
        self.env.apply_event(event)
        self.prefix.update(json.dumps(event, sort_keys=True, separators=(',', ':')).encode() + b'\n')
        self.event_count += 1
        if typ == 'dahai':
            self.pending_discard = event['actor']
        elif typ not in META:
            self.pending_discard = None
        return samples


def extract(cfg):
    catalog = Path(cfg['catalog_index'])
    if digest(catalog) != cfg['catalog_index_sha256']:
        raise ValueError('Registered split catalog changed')
    database = sqlite3.connect('file:' + str(catalog.resolve()) + '?mode=ro', uri=True)
    held = {g for (g,) in database.execute("SELECT DISTINCT group_key FROM groups WHERE split!='train'")}
    allowed = {g for (g,) in database.execute("SELECT DISTINCT group_key FROM groups WHERE split='train'")}
    allowed = {g for g in allowed - held if g.startswith('replay:')}
    database.close()
    if not allowed:
        raise ValueError('No historically train-only replay groups')
    for file, checksum in cfg['archives'].items():
        if digest(Path(file)) != checksum:
            raise ValueError('Registered raw archive changed')
    out = Path(cfg['out'])
    out.mkdir(parents=True, exist_ok=False)
    counts = Counter(); types = Counter(); failures = []; groups = set(); seen = set()
    with (out / 'train.jsonl').open('w') as stream:
        for file in cfg['archives']:
            with tarfile.open(file, 'r|*') as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith('.mjai.json.zst'):
                        continue
                    match = Path(member.name).name.removesuffix('.mjai.json.zst')
                    if 'replay:' + match not in allowed or match in seen:
                        continue
                    seen.add(match)
                    try:
                        if member.size > 10_000_000:
                            raise ValueError('Oversized compressed replay')
                        with archive.extractfile(member) as compressed:
                            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                                raw = reader.read(10_000_001)
                        if len(raw) > 10_000_000:
                            raise ValueError('Oversized decompressed replay')
                        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
                        starts = [e for e in events if e['type'] == 'start_kyoku']
                        if not starts or any(len(e['tehais']) != 4 for e in starts):
                            raise ValueError('Complete four-player starts required')
                        replay = PrefixReplay(cfg['shadow_seed'])
                        samples = []
                        for event in events:
                            for request in replay.consume(event):
                                prefix = request.pop('_prefix')
                                request['_meta'] = dict(id=f'{match}/kan/{len(samples)}', group_id=match,
                                    source='raw_mjai_prefix', split='train', schema=SCHEMA,
                                    prefix_events=prefix['event_count'], prefix_sha256=prefix['sha256'],
                                    archive_member=member.name,
                                    raw_match_sha256=hashlib.sha256(raw).hexdigest())
                                samples.append(request)
                        if not events or events[-1]['type'] != 'end_game':
                            raise ValueError('Complete end_game required')
                    except Exception as error:
                        failures.append(dict(match=match, error=str(error)[:300]))
                        continue
                    for request in samples:
                        stream.write(json.dumps(request, ensure_ascii=False) + '\n')
                        q = request['questions']['action']
                        types[q['criteria'][q['label']]['type']] += 1
                    counts['matches_replayed'] += 1
                    counts['records'] += len(samples)
                    if samples:
                        groups.add(match)
                    write(out / 'status.json', dict(status='extracting_unadmitted_prefix_kan',
                        counts=dict(counts), failed_matches=len(failures)))
    for file, checksum in cfg['archives'].items():
        if digest(Path(file)) != checksum:
            raise ValueError('Raw archive changed during extraction')
    if digest(catalog) != cfg['catalog_index_sha256']:
        raise ValueError('Split catalog changed during extraction')
    provenance = dict(schema=SCHEMA, config=cfg, files={'train': digest(out / 'train.jsonl')},
        matches={'train': sorted(groups), 'validation': [], 'test': []}, counts={'train': counts['records']},
        action_types=dict(types), failures=failures, prefix_inputs_built_before_label_event=True,
        censored_passes_excluded=True, historical_held_groups_excluded=len(held),
        training_admitted=False, independent_source_and_prefix_audit_required=True,
        test_payloads_opened=False, raw_records_public=False)
    write(out / 'provenance.json', provenance)
    result = dict(status='expert_kan_prefix_extracted_pending_independent_audit', counts=dict(counts),
        action_types=dict(types), failed_matches=len(failures), training_admitted=False,
        provenance_sha256=digest(out / 'provenance.json'))
    write(out / 'result.json', result); write(out / 'status.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    print(json.dumps(extract(json.loads(parser.parse_args().config.read_text()))))


if __name__ == '__main__':
    main()
