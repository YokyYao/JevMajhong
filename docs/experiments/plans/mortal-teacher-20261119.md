# Mortal teacher distillation from the audited176-step research checkpoint

The user authorizes training with the already downloaded
[VoidShine/mortal-298k](https://huggingface.co/VoidShine/mortal-298k) teacher.
Extend the ongoing pursuit/holding goal with this teacher phase. Formal Jev Play
v2/v8, main and the Windows release remain unchanged until qualification.

## Teacher and environment

Weights SHA256 bfb3a6c072aa0bfd4171a9cdc77cb6c02ae42cde920843f9e5784394f23447d8;
[Mortal](https://github.com/Equim-chan/Mortal) source
0cff2b52982be5b1163aa9a62fb01f03ce91e0d2. Use the previously audited Python3.12
libriichi binary from [Akagi-MjaiBot-Mortal](https://github.com/shinkuan/Akagi-MjaiBot-Mortal)
commit9b3936b3f32cab4c4f74330d1891fc4b069bee11, binary SHA256
3aeb09112198f6babe9bba4dd392ea009669ee7ba5d33cd3cd4242c737bb004b.
Both dependencies identify AGPL-3.0; preserve attribution and license when
redistributing their code. This repository references local dependencies and
does not redistribute their weights or binary in this experiment.

The third-party teacher was trained on hanchan with rank points[6,4,2,0].
Its reported similarity/MAKA grades are not independent strength evidence.
An existing retrospective384-match comparison reports teacher top2 .763021
versus releasedv8 .643229 against the same rule-opponent grid. These inspected
walls are not fresh confirmation, and this comparison is not against176.
That teacher selected zero daiminkan in166 available windows: do not teach
that pass is universally correct or claim kan solved by teacher imitation.

Use the existing RiichiEnv0.4.11 Mahjong Soul four-player red East environment.
Replay ONLY the acting player's public events into Mortal. Reject opponent
concealed tiles/draws. Load strictly with weights_only=True and greedy FP32
inference; disable the upstream hanchan-oriented rule-based agari guard.
Map chi/pon/riichi/ankan/daiminkan/kakan/pass including red tiles to the actual
legal menu; mapping failures stop without a heuristic fallback.

## New isolated sources and admission

384 TRAIN games starting2026140000;128 VALIDATION games starting2026145000.
Game identity is split before simulation and before labels; no reserved older
validation/independent/full-start or historical baseline game is training data.
Each game uses one controlled Mortal seat; opposing profiles alternate all
Mortal, two Mortal plus one calling rule, and one Mortal plus two rule opponents.
All models receive only their own observations. Full ledgers must reconcile.

Distill only East1--3, menus without a win, and teacher actions admitted by the
UNCHANGED current Jev serving guard. Record all excluded actions and reasons;
do not pretend a serving guard rejection was learned. East4/extension and
win-versus-pass decisions retain existing East-specific supervision.

Outcome-blind hash selection, at most16 selected menus per source game:

| Source | TRAIN | VALIDATION | Loss mass |
|---|---:|---:|---:|
|Ordinary teacher discards|512|128|25%|
|Teacher discards under declared-riichi threat|128|32|10%|
|Teacher riichi choices|64|16|5%|
|Teacher chi choices|64|16|7.5%|
|Teacher pon choices|64|16|7.5%|
|Teacher decline-call controls|128|32|10%|
|Kan windows, including choose/pass|64|16|5%|
|Existing ordinary/leading replay|608|unchanged old validation|15%|
|Existing endgame targets|266|unchanged old186 plus fresh8 monitoring|15%|

There are1024 new TRAIN and256 new VALIDATION menus, plus all874 existing
positive training rows. Kan selection must include all three kan types, each
with at least4 TRAIN and1 VALIDATION windows where available; missing coverage
stops before fitting rather than inventing examples. Report chosen-versus-pass
counts and per-type accuracy; no forced kan targets. Teacher targets are its
mapped action with2% uniform label smoothing, not calibrated win probabilities.
Only source-game-separated held-out menus measure distillation progress.

## Weight continuation, observation and selection

Warm-start EXACT audited176 LoRA AND head from targeted-refresh.176 is a research
candidate with fixed-menu primary .2351409314, not an independent proven model.
Keep the backbone frozen, update existing rank16 LoRA/head; no additional runtime
network. New AdamW moments,16 warmup updates, LR2e-6/1e-5, decay.001, clip1.
Each48-record accumulation covers all9 sources with fixed mass/quota coefficients.
At most512 NEW updates. Validate every32; observe at least192 new updates and
exposure to every positive record, then stop after four stale validations.
Small ordinary validation oscillations do not stop fitting during observation.
Nonfinite training stops immediately. Save optimizer and RNG with checkpoints.

EVERY validation saves weights and reloads the normal serving path, restoring
training RNG around temporary model loading. Report teacher CE/top1 by class,
old51/fresh6 finite-label top2, third pursuit, holding and replay CE separately.
Select the lowest teacher macro CE that improves at least2% relative to parent,
with old/fresh equally weighted top2 no worse than parent by2pp and holding
utility no worse by.05. These are screening tolerances, not success claims.
Progress uses macro CE with1e-4 relative improvement; equal hard choices do not
prematurely stop a still-improving distribution. Independently reload selected
and final checkpoints and require exact metric reproduction.

## Actual complete-match development comparison

After an eligible reload, run candidate and exact176 parent on8 new paired seed
groups starting2026149000, four controlled seats, two opponent profiles (one
calling plus two efficiency; one Mortal plus two efficiency):64 complete East
matches per arm. Teacher used as opponent is identical in both arms. Use the
same student formatter/action guard in both arms. Finish every native match,
validate ledgers and aggregate by whole seed, preserving correlation across
seats/profiles. Primary is candidate-minus-parent final top2; also report mean
rank, net points, hand win/deal-in rates and each call/kan type.

This128-match pilot is a DEVELOPMENT screen, not independent confirmation.
Positive top2 point estimate alone is weak evidence; whole-seed confidence
intervals must accompany it. If no positive top2 signal, keep the parent and
diagnose classes rather than increasing imitation loss blindly. If promising,
register a separate unused confirmation pool before release. Teacher agreement
alone never qualifies deployment. No automatic export/release.

All hashes, model loading, data generation, tests, fitting and evaluation run in
bounded Slurm GPU allocations with companion CPUs. Managed foreground stages,
timeouts, exit receipts and reaped children; no login computation or polling
daemon. Publish code/plan/aggregate findings only; games, traces and research
weights remain private. Never modify registered code/data while a job runs.
