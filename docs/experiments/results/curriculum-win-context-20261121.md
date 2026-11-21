# Public win-source input contract before stage4 labels

2026-11-21, Asia/Dubai. This is a native interface check, not a training or strength result. Current calls/attack-bound sources, earlier request contracts and the practical v8 policy remain unchanged.

The current native `Action.to_mjai()` represents both ron and tsumo as `hora` without the paying/source seat. The ordinary request removes actor/target as well. Endgame pursuit needs that information: ron against second place and self-draw have different score transfers. Using the most recent discard alone is also wrong for a robbed kan.

`tools/curriculum_win_context.py` adds a separate `public-win-source-v1` contract. It combines the exact native legal win kind with the current verified public event. An allowlist projects only trigger type, actor and public winning tile. Opponent draw tiles are removed. Native MJAI honor/red-five notation is normalized to the model's tile notation. The original action menu stays identical; the added state block maps each legal hora key to kind, source, source seat and tile. It never invents a payer from an older discard, reads another concealed hand or scores a hand.

Normal ron must match the source player's current river endpoint. Tsumo must match the observer's own current draw. A robbed-kan trigger is explicitly classified as `robbed_kakan` or `robbed_ankan`; unsupported or inconsistent triggers fail before data admission. Replay importers must provide a verified current public trigger rather than synthetic initialization history. This module is not enabled in existing fits or the Windows application, and does not claim screenshot integration.

GPU allocation312081, node ws-l1-011, ran the bounded native audit on diagnostic logical seeds2026400200–2026400207. These seeds were inspected beforehand to design the contract, so this is a development integration check. Reserved final seeds2026300000–2026300031 were not used. Four Efficiency agents completed each match with no learned model loaded.

| Actual native observation | Count |
|---|---:|
| Complete match ledgers checked |8|
| Exact native-menu/public-state parity checks |29|
| Ron / verified discard sources |20/20|
| Tsumo / verified own-draw sources |9/9|
| Honor winning tiles |3|
| Red winning tiles |0|
| Robbed-kan wins |0|

Six new regression checks cover mutation/menu preservation, normal source validation, robbed-added-kan versus stale-discard handling, self-draw/opponent draw redaction, malformed/unknown/self-ron rejection, concealed-kan public identity and MJAI tile normalization. All77 curriculum checks pass in allocated GPU resources. Red winning tiles and robbed kan currently have fixtures only; do not present them as observed native coverage. Internal ledger consistency is not independent han/fu verification.

```bash
python -m unittest discover -s tests/core -p 'test_curriculum*.py'
python tools/audit_curriculum_win_context.py \
  --registered runs/curriculum-optimization-v2/fit-plan/registered.json \
  --out runs/curriculum-optimization-v2/stage4-public-win-context-private.json
```

These commands require a bounded scheduler GPU allocation, including their companion CPU work. Use a new output path to repeat an audit. The new aggregate has no hidden hands, future worlds or per-window records.

Next establish public-information hidden-world sampling and consistent full-match continuations, then sufficient paired accept/decline labels across ranks, score gaps, dealer/extension, ron/tsumo and defensive contexts. The old hand-only targets cannot stand in for terminal top-two labels. No labels, optimizer updates, stage4 admission, terminal-policy improvement or v8 parity follow from this interface audit.
