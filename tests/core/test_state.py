import json
import base64
from pathlib import Path
import unittest

from riichienv import GameRule, RiichiEnv, Observation
from jevmahjong.core.state import build_request, describe, signature


class SemanticTests(unittest.TestCase):
    def test_training_and_play_use_same_observation_and_legal_schema(self):
        env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_mjsoul(), seed=911)
        obs = env.reset()[0]
        request, actions, feats = build_request(obs)
        self.assertEqual(len(request["questions"]["action"]["criteria"]), len(actions))
        self.assertEqual(len(set(signature(a) for a in actions)), len(actions))
        self.assertNotIn("hands", request["state"])
        self.assertNotIn("events", request["state"])
        self.assertNotIn("wall", request["state"])
        self.assertTrue(all(f["sh"] >= 0 for f in feats if f))
        before = json.dumps(request, sort_keys=True)
        # Synthetic replay events must not influence model state or visible tiles.
        encoded = json.loads(base64.b64decode(obs.serialize_to_base64()))
        encoded["events"] = [json.dumps({"type": "tsumo", "actor": 3, "pai": "1m"})]
        modified = Observation.deserialize_from_base64(base64.b64encode(json.dumps(encoded).encode()).decode())
        after, _, _ = build_request(modified)
        self.assertEqual(before, json.dumps(after, sort_keys=True))



    def test_concealed_opponent_hand_is_rejected(self):
        obs = RiichiEnv(seed=41).reset()[0]
        encoded = json.loads(base64.b64decode(obs.serialize_to_base64()))
        encoded["hands"][1] = [0, 4, 8]
        invalid = Observation.deserialize_from_base64(base64.b64encode(json.dumps(encoded).encode()).decode())
        with self.assertRaisesRegex(ValueError, "concealed hand"):
            build_request(invalid)


if __name__ == "__main__":
    unittest.main()
