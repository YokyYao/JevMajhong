"""Diagnose existing hybrid kan decisions on native scripted states, not strength."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from arena import EfficiencyAgent, guarded_indices
from call_policy import digest, is_call_request
from head_rl import write
from riichienv import ActionType, GameRule, RiichiEnv
from semantic import build_request, legal_candidates
from seed_control import engine_seed

KANS = ('ankan', 'daiminkan', 'kakan')


def generate(out, quota, max_games, seed_start):
    counts = Counter(); rows = []; baseline = EfficiencyAgent()
    started = time.monotonic()
    for offset in range(max_games):
        env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(),
            seed=engine_seed(seed_start + offset, 'spaced-v1'))
        observations = env.reset(); steps = 0
        while not env.done():
            selected = {}
            for seat, obs in observations.items():
                actions = legal_candidates(obs)
                kinds = [json.loads(a.to_mjai())['type'] for a in actions]
                wins = [a for a in actions if a.action_type in (ActionType.RON, ActionType.TSUMO)]
                if not wins and set(kinds) & set(KANS):
                    request, ordered, features = build_request(obs)
                    eligible = sorted(set(guarded_indices(obs, ordered, features)) |
                        {i for i, f in enumerate(features) if not f})
                    for kind in KANS:
                        if kind not in kinds or counts[kind] >= quota:
                            continue
                        indices = [i for i, c in enumerate(request['questions']['action']['criteria'].values()) if c['type'] == kind]
                        if not set(indices) <= set(eligible) or is_call_request(request):
                            raise ValueError('Kan dropped by guard or routed into chi/pon policy')
                        rows.append({'kan_type': kind, 'seed': seed_start + offset, 'seat': seat,
                            'step': steps, 'request': request, 'eligible': eligible,
                            'native_actions': [json.loads(a.to_mjai()) for a in ordered],
                            'native_kan_indices': indices})
                        counts[kind] += 1
                kans = [a for a, k in zip(actions, kinds) if k in KANS]
                calls = [a for a, k in zip(actions, kinds) if k in ('chi', 'pon')]
                selected[seat] = wins[0] if wins else kans[0] if kans else (
                    calls[0] if calls and (seat + offset) % 2 == 0 else baseline.act(obs))
            observations = env.step(selected); steps += 1
            if steps > 100000:
                raise RuntimeError('Scripted diagnostic failed to terminate')
        write(out / 'status.json', {'status': 'collecting_native_kan_cases',
            'scripted_games': offset + 1, 'cases': dict(counts), 'seconds': time.monotonic() - started})
        if all(counts[k] >= quota for k in KANS):
            break
    (out / 'native-cases-private.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    if not all(counts[k] >= quota for k in KANS):
        raise ValueError('Insufficient native kan cases: ' + str(dict(counts)))
    return rows


def infer(out, checkpoint, device):
    import inspect
    import torch
    torch.set_num_threads(4)
    if device == 'cpu':
        # This server has optional CUDA-only FLA installed. Use the upstream
        # reference functions in this process only, as the CPU Windows bundle does.
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        for name in ('torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
            setattr(qwen, name, inspect.unwrap(getattr(qwen, name)))
    from jevany import JevModel
    from jevany.checkpoint import LoadOptions
    from jevany.inference import InferenceOptions
    from jevany.api import validate_distribution
    from desktop.jev_sidecar.stage2_runtime import Stage2Client
    from encoder_call_policy import EncoderCallRouter
    base = Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    write(out / 'status.json', {'status': 'loading_existing_oldB', 'device': device})
    old = JevModel.from_pretrained(str(checkpoint), device=device, dtype='fp32' if device == 'cpu' else 'bf16',
        options=LoadOptions(merge=False, temperature=1., compile_mode=None, cuda_graphs=False,
            fused_kernels=False, base_load_path=str(base)),
        inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192,
            max_packed_tokens=8192, prefix_cache_size=0))

    class NoCall:
        def system_one(self, **request):
            raise ValueError('A kan request incorrectly reached the stage2 chi/pon branch')

    class Cached:
        def system_one(self, **request):
            if request != self.request:
                raise ValueError('Kan route changed the oldB payload')
            return self.response

    cases = [json.loads(l) for l in (out / 'native-cases-private.jsonl').read_text().splitlines()]
    summary = {k: {'native_legal_cases': 0, 'selected': Counter(), 'kan_preference_sum': 0.} for k in KANS}
    cache = Cached(); desktop = Stage2Client(cache, NoCall()); server = EncoderCallRouter(cache, NoCall())
    with (out / 'predictions-private.jsonl').open('w') as stream:
        for number, row in enumerate(cases, 1):
            request = row['request']; keys = list(request['questions']['action']['criteria'])
            output = old.system_one(**request)
            cache.request, cache.response = request, output
            if desktop.system_one(**request) != output or server.system_one(**request) != output:
                raise ValueError('Desktop/server hybrid differs from oldB on kan')
            p, _ = validate_distribution(output['answers']['action']['probabilities'], keys)
            chosen = max(row['eligible'], key=lambda i: p[i])
            kinds = [v['type'] for v in request['questions']['action']['criteria'].values()]
            record = {'kan_type': row['kan_type'], 'seed': row['seed'], 'seat': row['seat'],
                'step': row['step'], 'keys': keys, 'probabilities': p, 'chosen_type': kinds[chosen],
                'kan_preference': sum(p[i] for i in row['native_kan_indices'])}
            stream.write(json.dumps(record) + '\n'); stream.flush()
            item = summary[row['kan_type']]; item['native_legal_cases'] += 1
            item['selected'][kinds[chosen]] += 1; item['kan_preference_sum'] += record['kan_preference']
            write(out / 'status.json', {'status': 'testing_existing_oldB_kan', 'completed': number, 'total': len(cases)})
    result = {'status': 'complete_native_kan_diagnostic', 'device': device,
        'checkpoint': str(checkpoint.relative_to(ROOT)), 'head_sha256': digest(checkpoint / 'head.pt'),
        'cases_sha256': digest(out / 'native-cases-private.jsonl'),
        'predictions_sha256': digest(out / 'predictions-private.jsonl'),
        'summary': {k: {**v, 'selected': dict(v['selected']),
            'mean_kan_preference': v['kan_preference_sum'] / v['native_legal_cases']} for k, v in summary.items()},
        'server_and_desktop_routing_unchanged_to_oldB': True,
        'actual_screenshot_candidate_generation_not_covered': True,
        'scripted_cases_not_real_player_win_rate_or_kan_quality': True,
        'optimizer_updates': 0, 'human_test_opened': False}
    write(out / 'result.json', result); write(out / 'status.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--quota', type=int, default=12)
    parser.add_argument('--max-games', type=int, default=96)
    parser.add_argument('--seed-start', type=int, default=10500000)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--infer-only', action='store_true')
    args = parser.parse_args(); out = args.out.resolve()
    if not args.infer_only:
        out.mkdir(parents=True, exist_ok=False)
        generate(out, args.quota, args.max_games, args.seed_start)
    infer(out, ROOT / 'runs/head-rl-diagonal-value-20261030/checkpoints/calibration-02', args.device)


if __name__ == '__main__':
    main()
