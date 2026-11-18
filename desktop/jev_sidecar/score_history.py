"""Persist confirmed public score observations and hand-boundary settlements."""
import json
import os
from pathlib import Path
from threading import RLock
from time import time
from uuid import uuid4

from .endgame_policy import canonical_request
from jevmahjong.endgame.context import placements


class MatchScoreHistory:
    def __init__(self, path=None):
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local/share"))
        self.path = Path(path) if path is not None else local / "JevMahjong/score-history.jsonl"
        self.session_id = uuid4().hex
        self.match_id = uuid4().hex
        self.previous = None
        self.lock = RLock()

    def _append(self, event):
        event = dict(event, session_id=self.session_id, match_id=self.match_id, observed_at=time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")
        return event

    def observe(self, payload):
        screen = payload["state"].get("screen_observation", {})
        required = {"scores", "dealer", "round_wind", "hand_number", "honba", "sticks"}
        if not (screen.get("metadata_confirmed") or required <= set(screen.get("confirmed_metadata", []))):
            return None
        canonical, _ = canonical_request(payload)
        state = canonical["state"]
        scores, sticks = state["scores"], state["sticks"]
        if (len(scores) != 4 or any(not isinstance(s, int) for s in scores)
                or not isinstance(sticks, int) or sticks < 0 or sum(scores) + sticks * 1000 != 100000):
            return None
        marker = (state["round_wind"], state["hand_number"], state["honba"])
        snapshot = {"round_wind": marker[0], "hand_number": marker[1], "honba": marker[2],
                    "scores": list(scores), "placements": placements(scores), "sticks": sticks,
                    "own_original_seat": state["seat"], "dealer": state["dealer"],
                    "riichi": list(state["riichi"])}
        with self.lock:
            if self.previous is not None and self.previous["marker"] == marker:
                self.previous["latest"] = snapshot
                return None
            reset = marker == (0, 1, 0) and scores == [25000] * 4 and sticks == 0
            if self.previous is not None and reset:
                # A new match's reset scores are never treated as the previous final settlement.
                self._append({"kind": "match_reset", "previous_terminal_scores_observed": False})
                self.match_id = uuid4().hex
                self.previous = None
            if self.previous is not None:
                event = {"kind": "hand_boundary", "previous_hand": list(self.previous["marker"]),
                         "next_hand": list(marker), "next_public_scores": list(scores),
                         "next_public_placements": placements(scores), "sticks": sticks}
                if not state["riichi"]:
                    event.update(settlement_scores=list(scores),
                                 settlement_observed_at="next_hand_before_new_riichi_deposits",
                                 settlement_inferred=False)
                else:
                    event.update(settlement_scores=None,
                                 reason="new_hand_deposits_already_present_cannot_reconstruct_prior_settlement")
                self._append(event)
            self.previous = {"marker": marker, "initial": snapshot, "latest": snapshot}
            return self._append({"kind": "confirmed_hand_score_observation", **snapshot})

    def record_terminal_scores(self, scores, sticks=0):
        """For an explicit confirmed result observation, never from a new-match reset."""
        if len(scores) != 4 or sum(scores) + sticks * 1000 != 100000:
            raise ValueError("Terminal score observation does not conserve points")
        with self.lock:
            return self._append({"kind": "confirmed_terminal_scores", "scores": list(scores),
                                 "placements": placements(scores), "sticks": sticks})

