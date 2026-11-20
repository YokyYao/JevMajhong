# Public riichi context: diagnostic evidence and controlled fresh-base fit

2026-11-21, Asia/Dubai. This is a development experiment within the [four-course goal](../plans/curriculum-optimization-20261121.md). The practical v8 policy and active stable-v2 inputs remain unchanged. A diagnostic classifier is not a new Jev policy or evidence of improved game strength.

## Why test the representation

Stable-v2's validation riichi decisions still fluctuate: at292 updates it accepts13/37 teacher accepts and correctly declines47/70 teacher declines. Its plain exact agreement is46.81% and weighted CE1.442875. The matching16 development games yield top-two62.50%, net+187.50/match, hand-win18.62% and dealin16.21%, compared with frozen v8's62.50%,+3,612.50,22.37% and10.34% on the same grid. These are only two consumed development seed groups; a favorable top-two point does not establish comparable strength.

The original public request contains the full visible state, but all107 reach candidates have only `type`, while discard candidates contain shanten, ukeire and safety annotations. Weak discrimination in saved scores motivates an explicit representation experiment; it does not prove a causal feature deficit or incorrect expert labels.

## Data and public transform

`tools/curriculum_riichi_context.py` adds `public-riichi-context-v1` to reach candidates. It derives own seat/dealer/winds, scores and placement gaps, turn, opponent riichi, honba/sticks and summaries across all legal tenpai-preserving discards. Summaries include recorded ukeire, safety against declared opponents, visible dora/red material and value-honor roles. These are material/observation features, not a complete yaku/han/fu score or a promise that a hand is valuable. Dora is not itself a yaku. Unknown/not-applicable values remain null; no hidden ura, future result, selected expert action, label or source annotation enters the transform.

All9,292 TRAIN and1,972 validation rows remain once, with unchanged IDs, labels, groups and loss strata. Only466 TRAIN/107 validation reach windows gain the context. All8,826/1,865 ordinary windows remain unchanged. All reach windows have complete tenpai summaries. Null safety summaries mostly mean no declared opponent/no safe alternative, rather than failed feature extraction. Source groups remain570 TRAIN/163 validation, with675 separately reserved groups excluded.

The candidate data manifest binds original and enriched file hashes, feature/preparation code and group reservations. Preparation checks label/metadata counterfactual invariance and equal public training/inference views. Feature code SHA256: `232e2c9cfdc28827c071de6aceb977fba62169c2ce9a032febf1096784e59732`. Candidate data manifest SHA256: `f331d8c570f76a7995eeeb7366eab4bb7dd3615737deffa294efb8e3d68e7558`.

## Diagnostic probe, not a serving component

A fixed logistic probe uses20 context columns plus missingness flags, TRAIN-only normalization/imputation and all466 labeled TRAIN reach windows from300 groups. Seed2026112103,400 full-batch AdamW updates, learning rate0.03 and decay0.01 are fixed; there is no validation threshold or hyperparameter search. No Jev weights are loaded or updated and no classifier checkpoint is deployed.

On the already-consumed107 validation windows/74 groups, the fixed0.5 decision gives25 TP/12 FN/58 TN/12 FP: acceptance67.57%, correct decline82.86%, balanced accuracy75.21%, AUC0.779537. TRAIN BCE changes from0.681316 to0.535564. These are small development-set diagnostics, without population-strength claims. Probe losses are separate from Jev training curves. The result supports testing whether the same public representation helps the Jev learner; it does not satisfy a Jev phase gate.

## Controlled context-v3 protocol

The [registered configuration](../../../configs/training/curriculum-stage1-context-v3.json) starts again from the identical pinned Qwen3.5-0.8B base, with fresh rank16 LoRA/256-dimensional pointer. Keep stable-v2's initialization seed, rows/labels/groups, natural riichi ratio, loss masses, balanced without-replacement order, batch1/accumulation64, learning rates, dropout and fixed four epochs/584 updates. The only intended change is the versioned public input representation. Do not load the logistic probe or stack another corrective model.

Validate at0/73/.../584, play the same16 development matches at0/146/292/438 and64 at584. Reuse the frozen v8 reference cache. The sole candidate is final584, with the same declared stage-retention gate and the same complete-game risk/net/placement reporting. Neither validation-selected thresholds nor an intermediate peak may replace the registered candidate. This comparison remains development evidence; the full-route final seeds remain untouched.

Every saved checkpoint must carry an `input-contract.json` binding the model bundle, feature implementation and input schema. Serving applies one public transform followed by one unchanged model call; no action is forced by the transform. Extra prompt content may still affect latency. Plain v8 reference serving and its legal-win guard remain unchanged. A separate end audit checks all contracts, original-to-enriched data parity, reloaded predictions and native raw-public serving parity. The normal core audit still checks actual optimizer/data exposure, losses, complete games and settlement.

Registration does not admit later courses with the original plain input. Any child of a context-aware parent must explicitly use the same schema and feature-code identity, preserve replay and pass group exclusions before fitting. No stage2/3/4 fit or production release is authorized by a diagnostic result or passing software check.

Preflight311989 completed successfully on ws-l2-010:29 curriculum tests passed and all107 real-model public/enriched serving comparisons had maximum probability error0. A further7 parent-gate tests, including schema-mismatch rejection, passed in the active GPU allocation. Fresh-base context-v3 training311995 has started on ws-l2-010; audit311996 is queued with an `afterok` dependency and invalid-dependency cancellation. No trained context-v3 quality result exists at this point.

A bounded execution check subsequently observed stable-v2 at422 updates/26,904 records and context-v3 at37/2,368. All logged updates are contiguous/finite, both riichi classes occur in every update, and weighted per-source losses reconcile. Every bound input remains unchanged. Stable-v2's first two full epochs each consume exactly8,826/194/272 plain/accept/decline records. The controlled comparison also verifies all372 initial adapter tensors and4 initial head tensors exactly equal, the first-epoch order receipts identical, all original data/labels equal after the declared public transform, and the same reference cache reused. These checks establish execution fidelity, not trained decision quality. Keep initial/noisy transient game points separate from a learned improvement claim.

All preparation, tests, probe and model-path checks run in bounded scheduled GPU allocations with companion CPUs. Public artifacts contain code/configuration and aggregate descriptions; raw hands, predictions, diagnostic details and unreleased research weights remain private.
