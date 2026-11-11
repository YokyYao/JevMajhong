"""One command surface for maintained training, evaluation and data preparation."""
import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(prog='jevmahjong', add_help=False)
    parser.add_argument('command', choices=['collect', 'train', 'evaluate', 'prepare-data'])
    if len(sys.argv) == 1 or sys.argv[1] in ('-h', '--help'):
        parser.print_help(); return
    args, rest = parser.parse_known_args()
    modules = {'collect': 'training.collect', 'train': 'training.fit',
        'evaluate': 'evaluation.run', 'prepare-data': 'data.prepare'}
    sys.argv = [sys.argv[0], *rest]
    importlib.import_module('jevmahjong.' + modules[args.command]).main()

if __name__ == '__main__': main()
