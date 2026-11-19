# Mortal teacher: fresh-base training

Status: authorized and being prepared on 2026-11-20. This supersedes the earlier feasibility-only proposal. This experiment does not initialize from v8, audited176, or any other Mahjong-trained adapter or decision head.

## Initialization and supervision

Load the pinned Qwen/Qwen3.5-0.8B base (revision `2fc06364715b967f1860aea9cf38778875588b17`), initialize a new rank-16 LoRA and a new 256-dimensional pointer head. The backbone is frozen; this is fresh task adaptation, not full-parameter pretraining. No old Jev policy weights or v8 replay soft targets enter training.

Reuse the already collected Mortal-298k teacher actions: 22,321 training decisions from 384 complete source games. Select 512 validation decisions from 128 separate source games, before fitting. The entire source-game split stays disjoint. Targets are teacher choices with 2% label smoothing, not game outcomes. Inputs contain public observations and the controlled player's own hand. Mortal is a fixed local teacher, not an opponent whose parameters are trained.

## Sampling and loss

Classify by available opportunities before the selected action: ordinary discard, defense, riichi selected, riichi declined, chi/pon selected, chi/pon declined, and kan opportunities. A 48-example accumulated update samples 24/8/2/2/4/6/2 examples respectively. Total loss mass is 45% ordinary, 15% defense, 15% riichi, 20% chi/pon, and 5% kan. Within riichi and chi/pon, select/decline loss mass follows their actual training population frequency; balanced opportunity sampling must not create a fictitious 50% call or riichi prior.

The teacher pool contains no positive daiminkan examples. Preserve its 108 training and 30 validation decline opportunities; do not invent positive labels or claim this run teaches when to accept daiminkan. Validation explicitly includes all three kan opportunity types. Source collection excludes win menus and teacher choices blocked by the existing Jev serving guard; win decisions and omitted choices are outside this distillation scope. The guard is unchanged for the baseline comparison.

## Schedule and checkpoints

Maximum 1,536 optimizer updates; evaluate saved checkpoints every 64 updates. Observe at least 768 updates and expose every training decision before allowing early stopping. If a category needs more updates for full exposure, extend the observation floor to its next 64-update boundary. Stop only after four subsequent checks fail to improve weighted validation cross-entropy by 0.1%. Select the best validation checkpoint after the observation/exposure floor, rather than an underexposed early checkpoint. Save optimizer/RNG state and actual LoRA/head-change evidence.

AdamW: LoRA LR 2e-5, head LR 2e-4, 64-update warmup, cosine decay to 10%, weight decay .001, gradient norm cap 1. Both fresh LoRA and head must actually change. Validation uses the normal saved-model loader, temperature 1, no adapter merge. Report weighted cross-entropy plus per-opportunity accuracy and teacher/student select/decline counts. Teacher agreement is not a win-rate estimate.

## Match evaluation and reporting

After fitting, reload and audit the selected and final checkpoints. Run 64 complete native East games for the selected candidate: eight seeds, four controlled seats, and two opponent profiles (one calling rule agent or one Mortal, plus two efficiency rule agents). Reuse the independently audited 64 parent games only with the identical seeds, rules, formatter and serving guard, verifying source hashes and parent identity. Report top-two, first/fourth, net points, hand wins, deal-ins, action counts and a whole-seed paired uncertainty interval. This is a development comparison on a reused grid, not independent confirmation or a causal test of initialization alone.

The prior continuation experiment achieved 27/64 top-two finishes versus the parent's 31/64 and -1,706.25 versus -1,445.31 net points/game; it did not demonstrate improvement. It remains a comparison reference and does not supply initialization or targets for this run.

Run only within a bounded Slurm GPU allocation with companion CPUs, logs and exit receipts. Publish code, plans and aggregate metrics; keep source records, prediction detail and research checkpoints private. No automatic EXE update or release.

## Upstream attribution

Decision interface: [JevAny](https://github.com/SimpleJev/JevAny), pinned source recorded in the private run manifest. Teacher: [Mortal](https://github.com/Equim-chan/Mortal), AGPL-3.0; weights: [VoidShine/mortal-298k](https://huggingface.co/VoidShine/mortal-298k), SHA-256 `bfb3a6c072aa0bfd4171a9cdc77cb6c02ae42cde920843f9e5784394f23447d8`. Native teacher bridge provenance and applicable AGPL notices remain recorded in the preceding teacher plan and run manifests. No upstream weights are republished by this plan.
