"""Preserve current visible winning tile and discarder before legacy formatting."""
import base64
import copy
import json
from riichienv import ActionType
from jevmahjong.core.state import tile_name


def add_native_context(request, obs, actions):
    payload = copy.deepcopy(request)
    data = obs.to_dict()
    serialized = json.loads(base64.b64decode(obs.serialize_to_base64()))
    win = next((a for a in actions if a.action_type in (ActionType.RON, ActionType.TSUMO)), None)
    reacting = any(a.action_type in (ActionType.CHI, ActionType.PON, ActionType.DAIMINKAN, ActionType.RON)
                   for a in actions)
    if not win and not reacting:
        return payload
    screen = payload["state"].setdefault("screen_observation", {})
    if win:
        screen["winning_tile"] = tile_name(win.tile)
    if reacting:
        tid = win.tile if win and win.action_type == ActionType.RON else serialized.get("last_discard")
        # RiichiEnv 0.4.11 serializes last_discard as a physical tile ID,
        # without its actor. Each river's physical IDs are also public.
        actors = [p for p, river in enumerate(data["discards"]) if river and river[-1] == tid]
        if len(actors) == 1 and actors[0] != payload["state"]["seat"]:
            screen["reaction"] = {"seat": actors[0], "tile": tile_name(tid)}
    return payload
