"""Recommend a legal MJAI action for an exported RiichiEnv player observation."""
import argparse
import json
from pathlib import Path

from riichienv import Observation
from arena import EfficiencyAgent
from native_policy import NativeModelAgent as ModelAgent
from semantic import describe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("observation", type=Path, help="File containing serialize_to_base64() output")
    parser.add_argument("--checkpoint", help="Explicit experimental model, overrides current champion")
    parser.add_argument("--guarded", action="store_true")
    parser.add_argument("--champion", type=Path, default=Path("runs/champion.json"))
    args = parser.parse_args()
    checkpoint, guarded = args.checkpoint, args.guarded
    if not checkpoint and args.champion.exists():
        champion = json.loads(args.champion.read_text())
        if champion["active_policy"] == "model":
            checkpoint, guarded = champion["selected"]["checkpoint"], champion["selected"]["guarded"]
    obs = Observation.deserialize_from_base64(args.observation.read_text().strip())
    describe(obs)  # reject concealed-opponent information before any policy sees it
    agent = ModelAgent(checkpoint, guarded) if checkpoint else EfficiencyAgent()
    action = agent.act(obs)
    print(json.dumps({"action": json.loads(action.to_mjai()),
                      "policy": "model" if checkpoint else "efficiency_baseline",
                      "checkpoint": checkpoint, "guarded": guarded}, ensure_ascii=False))


if __name__ == "__main__":
    main()
