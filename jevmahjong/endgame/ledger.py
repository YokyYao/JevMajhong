"""Record every hand's public score settlement from native events."""
from .context import placements


def hand_score_history(log):
    scores, sticks = [25000] * 4, 0
    rows = []
    current = None

    def close():
        nonlocal current
        if current is not None:
            current.update(scores_after=list(scores), placements_after=placements(scores),
                           sticks_after=sticks)
            rows.append(current)
            current = None

    for event in log:
        kind = event["type"]
        if kind == "start_kyoku":
            close()
            if event["scores"] != scores or event["kyotaku"] != sticks:
                raise ValueError("Hand boundary scores or stick pool do not reconcile")
            current = {"bakaze": event["bakaze"], "kyoku": event["kyoku"],
                "honba": event["honba"], "dealer": event["oya"],
                "scores_before": list(scores), "placements_before": placements(scores),
                "sticks_before": sticks, "settlements": []}
        elif kind == "reach_accepted":
            scores[event["actor"]] -= 1000
            sticks += 1
        elif kind in ("hora", "ryukyoku"):
            if current is None:
                raise ValueError("Settlement without a current hand")
            deltas = event["deltas"]
            if len(deltas) != 4 or sum(deltas) != (sticks * 1000 if kind == "hora" else 0):
                raise ValueError("Settlement does not conserve points and stick pool")
            scores = [s + d for s, d in zip(scores, deltas)]
            if kind == "hora":
                sticks = 0
            current["settlements"].append({k: event[k] for k in
                ("type", "actor", "target", "deltas") if k in event})
        elif kind == "end_kyoku":
            close()
        elif kind == "end_game":
            close()
            if sticks:
                leader = min(range(4), key=lambda p: (-scores[p], p))
                scores[leader] += sticks * 1000
                sticks = 0
        if sum(scores) + sticks * 1000 != 100000:
            raise ValueError("Score history lost points")
    close()
    return {"hands": rows, "final_scores": scores, "final_ranks": placements(scores),
            "unclaimed_sticks": sticks}

