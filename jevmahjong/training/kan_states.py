"""Capture every native kan window under a fixed calling continuation."""
import argparse
import json
from jevmahjong.core.state import build_request
from jevmahjong.data.native_states import KAN_METHOD
from jevmahjong.policy.kan import is_kan_request
from .collector import NativeCollector
from .defense_states import collect_native_inputs


class KanStateCollector(NativeCollector):
    def act(self, obs, frozen=False, stochastic=False, trace=None):
        target = trace is not None and is_kan_request(build_request(obs)[0])
        return super().act(obs, frozen, stochastic, trace if target else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-walls', type=int)
    args = parser.parse_args()
    from pathlib import Path
    collect_native_inputs(json.loads(Path(args.config).read_text()), args.resume, args.max_walls,
                          collector=KanStateCollector, method=KAN_METHOD, scope='ordinary')


if __name__ == '__main__':
    main()
