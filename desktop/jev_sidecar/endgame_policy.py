"""Portable endgame input mapping and one-model serving for confirmed public state."""
import copy
import hashlib
import json
from pathlib import Path

MODEL_ID = "jev-endgame-placement-v1"
MODEL_NAME = "Jev Endgame v1"
IDENTITY_FILE = "endgame-identity.json"


def confirmed_for_endgame(payload):
    state = payload["state"]
    screen = state.get("screen_observation", {})
    required = {"scores", "dealer", "round_wind", "hand_number", "honba", "sticks"}
    known = bool(screen.get("metadata_confirmed")) or required <= set(screen.get("confirmed_metadata", []))
    scores, sticks = state.get("scores"), state.get("sticks")
    coherent = (isinstance(scores, list) and len(scores) == 4
                and all(isinstance(x, int) for x in scores)
                and isinstance(sticks, int) and sticks >= 0
                and sum(scores) + sticks * 1000 == 100000)
    return known and coherent and (state.get("round_wind", 0) > 0 or state.get("hand_number") == 4)


def canonical_request(request):
    """Map relative table seats back to starting-East order, preserving option keys."""
    payload = copy.deepcopy(request)
    state = payload["state"]
    hand, dealer = state.get("hand_number"), state.get("dealer")
    if not isinstance(hand, int) or not 1 <= hand <= 4 or dealer not in range(4):
        raise ValueError("A valid hand number and dealer are required")
    first_dealer = (dealer - (hand - 1)) % 4
    order = [(first_dealer + i) % 4 for i in range(4)]
    mapping = {relative: canonical for canonical, relative in enumerate(order)}
    state["seat"] = mapping[state["seat"]]
    state["dealer"] = mapping[dealer]
    for key in ("scores", "rivers", "melds", "tsumogiri_flags", "riichi_sutehais", "last_tedashis"):
        values = state.get(key)
        if isinstance(values, list) and len(values) == 4:
            state[key] = [values[i] for i in order]
    state["riichi"] = sorted(mapping[p] for p in state.get("riichi", []))
    last = state.get("last_discard")
    if isinstance(last, list) and len(last) == 2 and isinstance(last[0], int):
        last[0] = mapping[last[0]]
    elif isinstance(last, dict):
        for key in ("actor", "target", "seat"):
            if isinstance(last.get(key), int):
                last[key] = mapping[last[key]]
    reaction = state.get("screen_observation", {}).get("reaction")
    if isinstance(reaction, dict) and isinstance(reaction.get("seat"), int):
        reaction["seat"] = mapping[reaction["seat"]]
    for criterion in payload["questions"]["action"]["criteria"].values():
        if isinstance(criterion.get("safe"), list):
            criterion["safe"] = sorted(mapping[p] for p in criterion["safe"])
        for key in ("actor", "target"):
            if isinstance(criterion.get(key), int):
                criterion[key] = mapping[criterion[key]]
    return payload, mapping


def validate_checkpoint(directory):
    """Validate a distinct exported profile; never relabel the v2 bundle."""
    directory = Path(directory)
    profile = json.loads((directory / IDENTITY_FILE).read_text(encoding="utf-8"))
    from jevmahjong.endgame.context import INSTRUCTION
    from .unified_policy import identity
    released = identity()
    if (profile.get("schema") != "endgame-checkpoint-v1"
            or profile.get("model_id") != MODEL_ID
            or profile.get("input_version") != "endgame-public-v2"
            or profile.get("policy_instruction") != INSTRUCTION
            or profile.get("baseline_source_identity") != released["source_identity_sha256"]):
        raise ValueError("Endgame checkpoint/profile identity differs")
    required = {"adapter_model.safetensors", "adapter_config.json", "head.pt",
                "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
    files = profile.get("files", {})
    if set(files) != required:
        raise ValueError("Endgame inference bundle is incomplete")
    for name, checksum in files.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != checksum:
            raise ValueError("Endgame checkpoint file differs: " + name)
    import torch
    meta = torch.load(directory / "head.pt", map_location="cpu", weights_only=True)
    expected = released["runtime"]
    for key in ("base", "base_revision", "lora", "head_dim", "head_residual_dim",
                "head_type", "decision_mode", "branch_mode", "backbone_adapter"):
        if meta.get(key) != expected[key]:
            raise ValueError("Endgame runtime differs: " + key)
    return profile


class EndgamePolicyClient:
    def __init__(self, model):
        self.model = model

    def allow_win_pass(self, payload):
        return confirmed_for_endgame(payload)

    def system_one_with_trace(self, **request):
        from jevmahjong.endgame.context import model_request
        canonical, mapping = canonical_request(request)
        payload = model_request(canonical)
        trace = {"version": "endgame-public-v2", "route": "endgame",
                 "model_id": MODEL_ID, "relative_to_starting_east": mapping,
                 "source_request": copy.deepcopy(request), "model_request": copy.deepcopy(payload)}
        return self.model.system_one(**payload), trace

    def system_one(self, **request):
        return self.system_one_with_trace(**request)[0]

