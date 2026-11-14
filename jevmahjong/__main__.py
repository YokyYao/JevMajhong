"""One command surface for maintained training, evaluation and data preparation."""
import argparse
import importlib
import sys


def main():
    parser = argparse.ArgumentParser(prog='jevmahjong', add_help=False)
    parser.add_argument('command', choices=['collect', 'recover-closed-states', 'collect-defense-states', 'collect-call-states', 'collect-kan-states', 'collect-ev', 'audit-defense-states', 'audit-native-states', 'audit-futures', 'audit-context-fit', 'audit-call-head', 'train', 'train-context', 'train-call-head', 'train-base-lora', 'evaluate', 'evaluate-unified', 'audit-unified', 'prepare-data', 'prepare-base-lora-pool', 'prepare-base-lora-mix', 'prepare-prefix-kan', 'audit-prefix-kan', 'reserve-prefix-validation'])
    if len(sys.argv) == 1 or sys.argv[1] in ('-h', '--help'):
        parser.print_help(); return
    args, rest = parser.parse_known_args()
    modules = {'collect': 'training.collect', 'collect-defense-states':'training.defense_states', 'collect-call-states':'training.call_states', 'collect-kan-states':'training.kan_states', 'collect-ev':'training.ev_collect', 'train': 'training.fit',
        'train-context':'training.context_fit', 'train-call-head':'training.call_head', 'train-base-lora':'training.base_lora', 'audit-call-head':'evaluation.call_head_audit',
        'evaluate': 'evaluation.run', 'evaluate-unified':'evaluation.unified_run', 'audit-unified':'evaluation.unified_audit', 'audit-context-fit':'evaluation.context_audit', 'audit-defense-states':'evaluation.native_state_audit', 'audit-native-states':'evaluation.native_state_audit', 'audit-futures':'evaluation.future_audit', 'prepare-data': 'data.prepare', 'prepare-base-lora-pool':'data.base_lora_pool', 'prepare-base-lora-mix':'data.base_lora_mix', 'prepare-prefix-kan':'data.replay_prefix', 'audit-prefix-kan':'data.replay_prefix_audit', 'reserve-prefix-validation':'data.reserve_validation'}
    modules['recover-closed-states']='training.closed_states'
    sys.argv = [sys.argv[0], *rest]
    importlib.import_module('jevmahjong.' + modules[args.command]).main()

if __name__ == '__main__': main()
