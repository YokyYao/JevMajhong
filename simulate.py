"""Run four random agents under the simulator's Mahjong Soul rule preset."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=2)
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be positive")
    from riichienv import GameRule, RiichiEnv
    from riichienv.agents import RandomAgent

    for game in range(args.games):
        env = RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_mjsoul())
        agents = {seat: RandomAgent() for seat in range(4)}
        observations = env.reset()
        steps = 0
        while not env.done():
            actions = {}
            for seat, observation in observations.items():
                action = agents[seat].act(observation)
                if action.to_mjai() not in [a.to_mjai() for a in observation.legal_actions()]:
                    raise RuntimeError("Agent returned an illegal action")
                actions[seat] = action
            observations = env.step(actions)
            steps += 1
            if steps > 100000:
                raise RuntimeError("Game exceeded smoke-test step limit")
        print(json.dumps({"game": game, "steps": steps,
                          "scores": list(env.scores()), "ranks": list(env.ranks())}))


if __name__ == "__main__":
    main()
