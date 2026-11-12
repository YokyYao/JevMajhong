"""One command surface for maintained training, evaluation and data preparation."""
import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(prog='jevmahjong', add_help=False)
    parser.add_argument('command', choices=['collect', 'collect-defense-states', 'collect-ev', 'audit-defense-states', 'audit-futures', 'audit-context-fit', 'audit-call-head', 'train', 'train-context', 'train-call-head', 'evaluate', 'prepare-data'])
    if len(sys.argv) == 1 or sys.argv[1] in ('-h', '--help'):
        parser.print_help(); return
    args, rest = parser.parse_known_args()
    modules = {'collect': 'training.collect', 'collect-defense-states':'training.defense_states', 'collect-ev':'training.ev_collect', 'train': 'training.fit',
        'train-context':'training.context_fit', 'train-call-head':'training.call_head', 'audit-call-head':'evaluation.call_head_audit',
        'evaluate': 'evaluation.run', 'audit-context-fit':'evaluation.context_audit', 'audit-defense-states':'evaluation.native_state_audit', 'audit-futures':'evaluation.future_audit', 'prepare-data': 'data.prepare'}
    sys.argv = [sys.argv[0], *rest]
    importlib.import_module('jevmahjong.' + modules[args.command]).main()

if __name__ == '__main__': main()
