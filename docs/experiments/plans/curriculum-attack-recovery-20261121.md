# Risk-recovery experiment after the failed calls quality target

2026-11-21, Asia/Dubai. This is a **new experimental transfer**, declared after inspecting the completed calls result and before recovery updates. It is not the original qualified-parent route, and is not a claim that calls-final324 passed. The old gate, failed attempt312276, cancelled audit312277, data and bound source remain unchanged.

## Question and evidence

The [completed calls course](../results/curriculum-stage2-calls-20261121.md) improved five-call menu agreement by12.74 points and reduced dealin, but its full64 net/top-two means fell below the pre-calls stage1 parent. The whole-seed interval against that parent includes zero. Daiminkan remains overly selected on expert-decline menus; the model makes16 in64 games versus v8's4. Chi is already close to v8's count. Merely increasing all calls or selecting a favorable16-game checkpoint would not address this evidence.

The prepared attack objective was designed before those final games, but its original transfer guard rejected the calls result. This experiment asks whether introducing its conditional own-win/speed/dealin targets recovers complete-game utility after expert-only calling supervision. It does not assume that failure is just insufficient steps or absent negative examples.

## Frozen transfer and recipe

Run ID: `jev-curriculum-stage3-attack-recovery-v2`. The [experiment configuration](../../../configs/training/curriculum-stage3-attack-recovery-v2.json) keeps the original attack numerical recipe unchanged. Its `stage3-attack-v1` field identifies that recipe version; the run ID, paths and admission method distinguish this experiment.

The [separate policy](../../../configs/training/curriculum-calls-recovery-policy-v1.json) permits only this recorded final324 parent and only its net-gain failure. Original replay/retention, call improvement, dealin and complete software-evidence requirements remain. The seal, registration and final audit retain `original_calls_quality_passed=false`; `admitted_for_experiment` is a separate field. A scoped loader uses the frozen structural checks under a distinct method and restores all hooks. It does not modify the original verifier's files or make strict qualification pass.

Parent: fixed calls final324, identity `136a230dc227356aa7c8e95031210414a4078dacf7b4d38c4e03cd8cf71ef016`; no minimum-loss or small-grid selection. Continue the same rank16 LoRA/256-dimensional pointer on frozen Qwen3.5-0.8B, with a new optimizer. This extends the fresh-base route; it is not a v8 initialization or another stacked network.

| Setting | Value |
|---|---|
| TRAIN / validation decisions |22,959 /5,905|
| Exact expert replay retained |20,237 /4,721|
| Expert / masked simulation loss mass |75% /25%|
| Simulation phase mix |Equal closed-discard/open-discard/call/kan mass|
| Simulation reward |Own win +0.25 discounted speed −1.5 dealin; net coefficient0|
| Complete epochs / actual exposures |2 /45,918|
| Accumulation / optimizer updates |127 /362|
| LoRA / head peak learning rate |1e-5 /5e-5|
| Validation / games |0,181,362;16,16,64 complete matches|
| Sole candidate |Final362; cumulative1270|

Keep the exact prepared data and canonical masked targets. Unobserved alternatives contribute no target gradient. Original16/32-world hand-only labels and their uncertainty remain limitations; this is neither new self-play nor terminal-placement supervision. The [data source audit](../results/curriculum-stage3-provenance-20261121.md) and [token/reference diagnostic](../results/curriculum-calls-data-diagnostic-20261121.md) remain applicable because inputs and numerical loss are unchanged.

## Success and failure reporting

The original attack quality targets still apply: retain ordinary/riichi/call agreement, improve equal-phase MC CE by at least0.02, nonnegative final64 net gain over calls parent, and no dealin increase. Report separate choose/decline counts against the cached v8 diagnostic; aggregate agreement must not hide daiminkan errors.

Additionally require final64 net and top-two means at least the **pre-calls stage1** parent, hand-win at least the calls parent and dealin no higher than the calls parent. Both original attack and additional recovery targets must pass. Report whole-seed paired95% intervals against calls, pre-calls stage1 and frozen v8; the earlier cache is source-bound and was already audited. These are development targets, not noninferiority evidence or permission to deploy.

No stage4 auto-launch or Windows replacement. Preserve failures and complete exposure; do not promote an intermediate peak. The full four-course objective and untouched final256-match/policy comparison to v8 remain unchanged. Reusing these development seeds is explicitly tuning, not an independent replication success.

## Actual readiness and execution

GPU preparation312297 on ws-l4-004 passed91 curriculum regressions and verified the actual audited failed parent, exact replay/group ancestry and pinned `bind_parent`. Registration and169-file launch binding were written before the first update. Source99731a98 passed core/desktop CI44854563641 before submission. Training312301 is now running on ws-l2-010, with final audit312304 afterok fit. Actual initialization verifies all372 adapter/4 head tensors exactly, frozen base, new optimizer and zero ancestral validation overlap. It is in baseline validation at this observation; real optimizer changes and policy effects remain pending.

New entry points are `tools/register_curriculum_attack_recovery.py`, `tools/train_curriculum_attack_recovery.py` and `tools/audit_curriculum_attack_recovery.py`. Use `seal_recovery` in `tools/curriculum_calls_recovery_parent.py` to create a fresh explicit experimental seal from the completed calls registration, audit, fit/audit exit receipts, captured scheduler journal and unchanged attack data manifest. Then register inside the GPU allocation:

```bash
python tools/register_curriculum_attack_recovery.py \
  --config configs/training/curriculum-stage3-attack-recovery-v2.json \
  --parent-seal "$EXPERIMENTAL_CALLS_SEAL" \
  --operation-diagnostic runs/curriculum-stage3-attack-v1/calls-data-audit-v1/summary-private.json
```

Before submitting, bind the registration inputs and actual launchers in the run root's `launch-code-binding-private.json`; bind that file plus the audit inputs in `audit-binding-private.json`. The trainer refuses missing/changed launch bindings. Set `JEV_ATTACK_REGISTERED`, `JEV_ATTACK_AUDIT_BINDING` and `JEV_ATTACK_EVIDENCE`, then use `cluster/curriculum-stage3-recovery.slurm` and `cluster/curriculum-stage3-recovery-audit.slurm`, with audit afterok fit and invalid dependencies cancelled. The bounded templates request GPU1/CPU4/48GB and6h/45min, with logs, subprocess timeouts and exit receipts. No login computation or background poller is used.

The policy's parent identity is specific to this audited experiment. A different-host replication must record its own actual final identity/results and declare any experimental admission before its child updates; it must not fake this identity or overwrite a failed strict grade. All new raw hands, worlds, prediction/confidence/order details and weights remain private. Source, plans, aggregate results and actual curves are public in the research PR.
