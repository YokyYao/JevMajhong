import json
from pathlib import Path
import tempfile
import unittest

from desktop.jev_sidecar.score_history import MatchScoreHistory


def payload(hand=3, scores=None, sticks=0, riichi=None, confirmed=True):
    return {"state": {"seat": 2, "dealer": hand - 1, "round_wind": 0,
        "hand_number": hand, "honba": 0, "scores": scores or [32000, 25000, 22000, 21000],
        "sticks": sticks, "riichi": riichi or [],
        "screen_observation": {"metadata_confirmed": confirmed}},
        "questions": {"action": {"type": "choice", "criteria": {}}}}


class ScoreHistoryTests(unittest.TestCase):
    def test_explicit_final_scores_rotate_ties_and_deduplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'scores.jsonl'
            history = MatchScoreHistory(file)
            final = payload(4, [25000, 22000, 25000, 28000])
            final['state']['dealer'] = 1
            event = history.record_terminal_observation(final)
            self.assertEqual(event['scores'], [25000, 28000, 25000, 22000])
            self.assertEqual(event['own_original_seat'], 0)
            self.assertEqual(event['placements'][0], 2)
            self.assertIsNone(history.record_terminal_observation(final))
            self.assertEqual(len(file.read_text().splitlines()), 1)

    def test_recorded_terminal_separates_next_match_even_after_riichi_deposit(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'scores.jsonl'
            history = MatchScoreHistory(file)
            history.observe(payload(4))
            history.record_terminal_observation(payload(4))
            history.observe(payload(1, [24000, 25000, 25000, 25000], sticks=1, riichi=[0]))
            rows = [json.loads(x) for x in file.read_text().splitlines()]
            self.assertIn('match_reset', [r['kind'] for r in rows])
            self.assertNotIn('hand_boundary', [r['kind'] for r in rows])
            self.assertNotEqual(rows[0]['match_id'], rows[-1]['match_id'])

    def test_recorded_terminal_does_not_join_a_later_first_seen_hand_to_old_match(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'scores.jsonl'
            history = MatchScoreHistory(file)
            history.observe(payload(4))
            history.record_terminal_observation(payload(4))
            history.observe(payload(2, [23000, 27000, 25000, 25000]))
            rows = [json.loads(x) for x in file.read_text().splitlines()]
            reset = next(r for r in rows if r['kind'] == 'match_reset')
            self.assertTrue(reset['previous_terminal_scores_observed'])
            self.assertNotIn('hand_boundary', [r['kind'] for r in rows])

    def test_final_result_rejects_missing_or_noninteger_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            history = MatchScoreHistory(Path(directory) / 'scores.jsonl')
            for scores in ([25000] * 3, [25000.] * 4, [25000, 25000, 25000, 24000]):
                with self.assertRaises(ValueError):
                    history.record_terminal_scores(scores)

    def test_records_real_next_hand_scores_once(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "scores.jsonl"
            history = MatchScoreHistory(file)
            history.observe(payload())
            result = payload(4, [32000, 23000, 24000, 21000])
            history.observe(result)
            history.observe(result)
            rows = [json.loads(x) for x in file.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            boundary = next(r for r in rows if r["kind"] == "hand_boundary")
            self.assertEqual(boundary["settlement_scores"], [32000, 23000, 24000, 21000])
            self.assertEqual(boundary["next_public_placements"][2], 2)

    def test_does_not_invent_prior_scores_after_new_deposits(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "scores.jsonl"
            history = MatchScoreHistory(file)
            history.observe(payload())
            history.observe(payload(4, [32000, 22000, 24000, 21000], sticks=1, riichi=[1]))
            rows = [json.loads(x) for x in file.read_text().splitlines()]
            boundary = next(r for r in rows if r["kind"] == "hand_boundary")
            self.assertIsNone(boundary["settlement_scores"])
            self.assertIn("deposits", boundary["reason"])

    def test_new_match_reset_is_not_a_previous_terminal_settlement(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "scores.jsonl"
            history = MatchScoreHistory(file)
            history.observe(payload(4))
            history.observe(payload(1, [25000] * 4))
            rows = [json.loads(x) for x in file.read_text().splitlines()]
            self.assertIn("match_reset", [r["kind"] for r in rows])
            self.assertNotIn("hand_boundary", [r["kind"] for r in rows])
            self.assertNotEqual(rows[0]["match_id"], rows[-1]["match_id"])

    def test_unconfirmed_state_is_not_saved_as_a_score_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "scores.jsonl"
            history = MatchScoreHistory(file)
            self.assertIsNone(history.observe(payload(confirmed=False)))
            self.assertFalse(file.exists())


if __name__ == "__main__":
    unittest.main()

