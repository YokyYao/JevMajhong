"""Prefix reconstruction must match actual native decisions, without labels."""
import copy
from collections import Counter
import io
import json
from pathlib import Path
import sqlite3
import shutil
import tarfile
import tempfile
import unittest

from riichienv import RiichiEnv, GameRule
from riichienv.convert import tid_to_mpsz
import zstandard

from jevmahjong.core.io import digest
from jevmahjong.core.state import build_request, legal_candidates
from jevmahjong.data.replay_prefix import PrefixReplay, action_key, extract, KAN
from jevmahjong.data.replay_prefix_audit import audit
from jevmahjong.data.base_lora_pool import verified_prefix_source
from jevmahjong.evaluation.agents import CallingOpponent, EfficiencyAgent
from tests.core.test_data_catalog import catalog


def normalized(request, label=True):
    req = copy.deepcopy(request)
    req.pop('_meta', None); req.pop('_prefix', None)
    state = req['state']
    if state.get('last_discard') is not None:
        state['last_discard'] = tid_to_mpsz(state['last_discard'])
    for key in ('last_tedashis', 'riichi_sutehais'):
        state[key] = [None if t is None else tid_to_mpsz(t) for t in state[key]]
    if not label:
        req['questions']['action'].pop('label', None)
    return json.dumps(req, sort_keys=True)


class ReplayPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.expected = Counter(); cls.emitted = Counter(); cls.routes = Counter()
        cls.first_events = None; cls.daiminkan = None
        cls.decline_response = None; cls.decline_self = None
        caller, closed = CallingOpponent(), EfficiencyAgent()
        for seed in range(882001, 882033):
            live = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=seed)
            observations = live.reset(); replay = PrefixReplay(seed + 300000); cursor = 0
            while True:
                events = live.mjai_log
                for event in events[cursor:]:
                    for req in replay.consume(event):
                        cls.emitted[normalized(req)] += 1
                        q = req['questions']['action']
                        kind = q['criteria'][q['label']]['type']
                        cls.routes[kind] += 1
                        if kind == 'daiminkan' and cls.daiminkan is None:
                            cls.daiminkan = (copy.deepcopy(events[:req['_prefix']['event_count']]), copy.deepcopy(event))
                cursor = len(events)
                if live.done():
                    break
                chosen = {}
                for player, obs in observations.items():
                    actions = legal_candidates(obs)
                    kinds = lambda a: json.loads(a.to_mjai())['type']
                    kans = [a for a in actions if kinds(a) in KAN]
                    if any(kinds(a) == 'daiminkan' for a in kans) and cls.decline_response is None:
                        cls.decline_response = (live.clone(), player, list(observations))
                    if any(kinds(a) in ('ankan', 'kakan') for a in kans) and cls.decline_self is None:
                        cls.decline_self = (live.clone(), player, list(observations))
                    wins = [a for a in actions if kinds(a) == 'hora']
                    passes = [a for a in actions if kinds(a) == 'none']
                    action = passes[0] if seed % 4 == 0 and player == 0 and wins and passes else (
                        kans[0] if kans else (caller if player % 2 else closed).act(obs))
                    chosen[player] = action
                    if any(kinds(a) in KAN for a in actions):
                        request, _, _ = build_request(obs)
                        criteria = request['questions']['action']['criteria']
                        key = action_key(action)
                        request['questions']['action']['label'] = next(k for k, v in criteria.items() if action_key(v) == key)
                        cls.expected[normalized(request)] += 1
                observations = live.step(chosen)
            if cls.first_events is None:
                cls.first_events = copy.deepcopy(events)

    def test_every_emitted_label_matches_an_actual_native_input_and_choice(self):
        self.assertFalse(self.emitted - self.expected)
        self.assertGreater(sum(self.emitted.values()), 30)
        for kind in KAN:
            self.assertGreater(self.routes[kind], 0)

    def check_declined(self, saved, response):
        env, player, active = saved
        env = env.clone(); observations = env.get_observations(active)
        request, _, _ = build_request(observations[player])
        chosen = {}
        for pid, obs in observations.items():
            kind = 'none' if response else 'dahai'
            chosen[pid] = next(a for a in legal_candidates(obs) if json.loads(a.to_mjai())['type'] == kind)
        target = action_key(chosen[player]); q = request['questions']['action']
        q['label'] = next(k for k, v in q['criteria'].items() if action_key(v) == target)
        replay = PrefixReplay(8123); events = env.mjai_log
        for event in events:
            replay.consume(event)
        cursor = len(events); env.step(chosen)
        emitted = [r for event in env.mjai_log[cursor:] for r in replay.consume(event)]
        matching = [r for r in emitted if r['state']['seat'] == player]
        self.assertEqual(len(matching), 1)
        self.assertEqual(normalized(matching[0]), normalized(request))

    def test_identifiable_draw_after_pass_preserves_negative_daiminkan_example(self):
        self.check_declined(self.decline_response, True)

    def test_declined_self_kan_preserves_actual_discard_example(self):
        self.check_declined(self.decline_self, False)

    def test_future_pon_vs_daiminkan_changes_label_but_not_public_input(self):
        events, actual = self.daiminkan
        first, second = PrefixReplay(71), PrefixReplay(93)
        for event in events:
            first.consume(event); second.consume(event)
        req = first.request(actual['actor'])
        pon = next(v for v in req['questions']['action']['criteria'].values() if v['type'] == 'pon')
        alternative = {k: v for k, v in actual.items() if k not in ('pai', 'consumed')}
        alternative.update(pon)
        a, b = first.consume(actual), second.consume(alternative)
        self.assertEqual(len(a), 1); self.assertEqual(len(b), 1)
        self.assertEqual(normalized(a[0], label=False), normalized(b[0], label=False))
        self.assertNotEqual(a[0]['questions']['action']['label'], b[0]['questions']['action']['label'])
        self.assertEqual(a[0]['_prefix'], b[0]['_prefix'])

    def fixture(self, root):
        from jevmahjong.training.base_lora import package_digest
        directory = root/'catalog'; directory.mkdir(); index = directory/'index-private.sqlite'
        db = sqlite3.connect(index); catalog.initialize(db)
        db.executemany('INSERT INTO groups VALUES (?,?,?)', [('replay:train','train',0), ('replay:held','train',0), ('replay:held','test',0)])
        db.execute("CREATE VIEW training_candidates AS SELECT e.* FROM entries e WHERE e.split='train' AND e.schema_valid=1 AND NOT EXISTS(SELECT 1 FROM groups g WHERE g.group_key=e.group_key AND g.split!='train')")
        db.commit(); db.close()
        (directory/'sources-private.json').write_text('[]')
        (directory/'result.json').write_text(json.dumps(dict(private_index_sha256=digest(index))))
        archive = root / 'raw.tar'
        raw = ('\n'.join(json.dumps(e) for e in self.first_events) + '\n').encode()
        with tarfile.open(archive, 'w') as tar:
            for name, data in [('held', b'invalid zstandard; must not be opened'),
                               ('train', zstandard.ZstdCompressor().compress(raw))]:
                item = tarfile.TarInfo(name + '.mjai.json.zst'); item.size = len(data)
                tar.addfile(item, io.BytesIO(data))
        cfg = dict(catalog_index=str(index), catalog_index_sha256=digest(index),
            archives={str(archive):digest(archive)}, out=str(root/'out'), shadow_seed=31)
        result = extract(cfg)
        source = root / 'source'; source.mkdir()
        shutil.copytree(Path(__file__).resolve().parents[2]/'jevmahjong', source/'jevmahjong',
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        launch = root/'launch.json'; launch.write_text(json.dumps(dict(command='prepare-prefix-kan',
            config=cfg, source=str(source), source_sha256=package_digest(source/'jevmahjong'))))
        qa = root/'native-qa.json'; qa.write_text(json.dumps(dict(mismatches=[],
            checks_shared_public_request_and_features=True, uses_different_shadow_random_seed=True,
            action_opportunities=dict(self.routes))))
        audit_cfg = dict(producer_out=str(root/'out'),
            producer_provenance_sha256=digest(root/'out/provenance.json'),
            producer_result_sha256=digest(root/'out/result.json'), producer_train_sha256=digest(root/'out/train.jsonl'),
            producer_launch=str(launch), producer_launch_sha256=digest(launch), native_probe=str(qa),
            native_probe_sha256=digest(qa), out=str(root/'audit'))
        return result, audit_cfg

    def test_extractor_never_opens_held_payload_and_never_self_admits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); result, audit_cfg = self.fixture(root)
            self.assertFalse(result['training_admitted']); self.assertEqual(result['failed_matches'], 0)
            self.assertEqual(result['counts']['matches_replayed'], 1)
            provenance = json.loads((root/'out/provenance.json').read_text())
            self.assertEqual(provenance['matches']['train'], ['train'])
            self.assertFalse(provenance['test_payloads_opened'])
            self.assertEqual(provenance['files']['train'], digest(root/'out/train.jsonl'))
            proof = audit(audit_cfg)
            self.assertEqual(proof['status'], 'passed_complete')
            self.assertEqual(proof['counts']['records'], result['counts']['records'])
            self.assertFalse(proof['training_started'])
            base = root/'catalog'; before = digest(base/'index-private.sqlite')
            cfg = dict(input_catalog=str(base),input_result_sha256=digest(base/'result.json'),
                input_sources_sha256=digest(base/'sources-private.json'),append_replay_source=str(root/'out'),
                audit_result=str(root/'audit/result.json'),audit_result_sha256=digest(root/'audit/result.json'),
                audit_protocol=str(root/'audit/protocol.json'),audit_protocol_sha256=digest(root/'audit/protocol.json'),
                out=str(root/'extended'))
            appended = catalog.append_replay(cfg)
            self.assertEqual(appended['appended_audited_records'], result['counts']['records'])
            self.assertEqual(appended['training_candidates_before_quality_selection']['rows'], result['counts']['records'])
            self.assertEqual(before, digest(base/'index-private.sqlite'))
            source = json.loads((root/'extended/sources-private.json').read_text())[0]
            inputs = {}; verified_prefix_source(source, provenance, inputs)
            self.assertIn(str(root/'audit/result.json'), inputs)
            with self.assertRaisesRegex(ValueError,'Complete observed prefix source audit required'):
                verified_prefix_source(dict(source,audit_receipts=[]),provenance,{})
            cfg.update(audit_result_sha256='0'*64,out=str(root/'bad'))
            with self.assertRaisesRegex(ValueError,'Bound catalog/source input changed'):
                catalog.append_replay(cfg)

    def corrupted_fixture(self, root, drop=False):
        _, cfg = self.fixture(root)
        train = root/'out/train.jsonl'; rows = [json.loads(l) for l in train.read_text().splitlines()]
        if drop:
            rows.pop(0)
        else:
            q = rows[0]['questions']['action']
            q['label'] = next(k for k in q['criteria'] if k != q['label'])
        train.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        provenance = root/'out/provenance.json'; p = json.loads(provenance.read_text())
        p['files']['train'] = digest(train); p['counts']['train'] = len(rows)
        provenance.write_text(json.dumps(p))
        result = root/'out/result.json'; r = json.loads(result.read_text())
        r['provenance_sha256'] = digest(provenance); r['counts']['records'] = len(rows)
        result.write_text(json.dumps(r))
        cfg.update(producer_train_sha256=digest(train), producer_provenance_sha256=digest(provenance),
            producer_result_sha256=digest(result))
        return cfg

    def test_independent_raw_audit_rejects_wrong_label_even_with_updated_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.corrupted_fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, 'Observed replay choice differs'):
                audit(cfg)

    def test_independent_raw_audit_rejects_omitted_window_even_with_updated_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self.corrupted_fixture(Path(tmp), drop=True)
            with self.assertRaisesRegex(ValueError, 'Missing observed prefix record'):
                audit(cfg)


if __name__ == '__main__':
    unittest.main()
