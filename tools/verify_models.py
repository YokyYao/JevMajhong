"""Verify named inference assets, optionally checking native cached TRAIN parity."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--discard', type=Path, default=ROOT / 'models/jev-discard-v1')
    parser.add_argument('--call', type=Path, default=ROOT / 'models/jev-call-v1')
    parser.add_argument('--native-pairs', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    from desktop.jev_sidecar.model_identity import validate_checkpoint
    from desktop.jev_sidecar.call_policy import validate_call_checkpoint
    result = {'model_id': 'jev-play-v1', 'discard': validate_checkpoint(args.discard),
        'call': validate_call_checkpoint(args.call), 'parameter_updates': 0}
    if args.native_pairs:
        import torch
        from jevmahjong.core.io import digest, request_hash
        from jevmahjong.policy.routing import HybridPolicy, load_model
        audit = json.loads(args.native_pairs.with_suffix('.audit.json').read_text())
        if digest(args.native_pairs) != audit['sha256']:
            raise ValueError('Native pair shard differs from its audit')
        rows = torch.load(args.native_pairs, map_location='cpu', weights_only=False)['rows']
        if args.device == 'cpu': torch.set_num_threads(4)
        ordinary = load_model(args.discard, args.device); calling = load_model(args.call, args.device)
        router = HybridPolicy(ordinary, calling); checked = {'ordinary': 0, 'calling': 0}
        seen = set(); maximum = 0.
        for row in rows:
            scope = row['scope']; identity = request_hash(row['public_request'])
            if checked[scope] >= 3 or identity in seen: continue
            seen.add(identity)
            p = router.system_one(**row['public_request'])['answers']['action']['probabilities']
            actual = torch.tensor(list(p.values())); error = float((actual - row['reference']).abs().max())
            maximum = max(maximum, error)
            if error > 2e-5:
                raise ValueError('Named runtime differs from the native source probabilities')
            checked[scope] += 1
        if any(n < 2 for n in checked.values()): raise ValueError('Need both policy scopes')
        result.update(native_probabilities_checked=checked, maximum_probability_error=maximum, device=args.device,
            native_pairs_sha256=audit['sha256'], human_test_opened=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

if __name__ == '__main__': main()
