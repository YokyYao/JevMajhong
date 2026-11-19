# Targeted fresh positions, paired64 labels and continuation from128

The user requests new targeted positions, more reliable labels, a revised mix,
training from the128-step model and publication of the plan. The duration-only
extension through384 failed: actual loaded fourth validation top2 .225490 at128
fell to .210172 at384 while TRAIN hits26→34/48. More repeated fitting was not
a generalization gain. This experiment changes source coverage and supervision.

## New outcome-blind sources

Generate1536 new TRAIN source games (logical seeds2026130000..2026131535)
and512 separate MONITOR source games (2026133000..2026133511), four-player
Mahjong Soul East rules, spaced-v1 engine seeds. These blocks are separate
from previous train/validation/development/independent/full-start blocks.
Rule source profiles vary closed/riichi, dama, calling and legal kan exploration;
they produce legal positions, not expert targets. Never fabricate tile arrays
or choose positions by their future score. Save full native logs and ledgers.

Select one position per distinct source game, balancing public deficit bands
(≤3000,3001–8000,>8000), dealer status and visible threats where available:

| Kind | New TRAIN menus | New MONITOR menus |
|---|---:|---:|
|Fourth discard/riichi choices|16|2|
|Fourth chi versus alternatives|4|1|
|Fourth pon versus alternatives|4|1|
|Fourth kan versus alternatives|4|1|
|Fourth small-win/continue public-gated choices|4|1|
|Third pursuit|8|1|
|Leading holding|8|1|
|Total|48|8|

Kan selection records ankan/daiminkan/kakan availability. Coverage shortfalls
stop before labels; do not pretend a quota was met or manufacture a correct
call. Both profitable and unprofitable calls remain because selection precedes
counterfactual outcomes. The four TRAIN kan menus include at least one legal
ankan, one daiminkan and one kakan from distinct actual games. Compare EVERY
legal action, not just a rule's favorite.
The old186 validation rows remain byte-identical; the8 new monitor rows are
held out from gradients. They are development monitoring, not independent proof.

## More reliable finite simulation labels

Use64 paired worlds per action, two independently seeded32-world blocks. Each
block balances two frozen hero continuation policies (formalv8 original and
the128-step endgame policy) and two opponent profiles (one/two calling opponents).
Within a world ALL actions share the same hidden wall, future-hand RNG and
initial simultaneous opponent actions. Resample unseen current wall and future
hand RNG; verify current observations/menus remain identical and future hands
actually vary in native tests. Current opponent concealed hands stay fixed to
the native source: this is NOT a full public-information posterior or expert
ground truth. New source diversity and future uncertainty address only part
of that limitation.

RiichiEnv's live environment has no writable seed. Future-hand resampling uses
its supported `reset(oya,round_wind,honba,kyotaku,scores,seed)` only AFTER a
native hand boundary. Verify the new dealer/kyoku/wind/honba/sticks/scores
exactly match, replace only the newly generated hand and stitch completed
MJAI prefixes into one reconciling ledger. Current hand/public history never
reset. Native tests require changed next deals and a complete valid match
ledger; an unsupported seed assignment cannot silently count as resampling.

Simulate until native match completion, rejecting truncation and any mismatch
between per-hand scores, final ledger and ranks. Keep per-policy/per-block
estimates and paired standard errors. Maximum768 legal actions across56 menus,
49,152 complete branch rollouts; two GPU labeling workers, bounded8h jobs.
Source/data code hashes and frozen policy identities are registered first.

Reward is final top2 (1 for ranks1/2,0 otherwise). Construct SOFT improvement
targets around the128-step action distribution, not a one-hot MC argmax:
paired advantage relative to its greedy eligible action is blended equally
between raw mean and SE-shrunk mean (z1.96 heuristic), then exponential tilt
temperature.1 and exploration.1. Record disagreement between32-world blocks;
do not delete rare fourth-to-top2 successes or all uncertain menus. If the
exploration target lowers finite expected top2, retain the supported prior.
SE shrinkage is not a confidence guarantee. Negative evidence is retained.

## Revised objective mass, not duplicated record percentages

| Loss source | Mass |
|---|---:|
|Old ordinary replay|30%|
|Old leading replay|15%|
|Old fourth labels|10%|
|Old third labels|2.5%|
|Old holding labels|2.5%|
|New fourth discards|13%|
|New fourth calls|12%|
|New fourth win/continue|5%|
|New third pursuit|5%|
|New holding|5%|

New data have40% total loss mass; old fourth MC targets drop from35% to10%.
Total fourth objective is40%, replay retention45%, third/holding15%. Each48
record accumulation has all ten sources, with per-record coefficient equal to
source mass/source batch quota. All existing826 positive rows remain and new48
bring the positive set to874. Old79 forced-win controls remain outside loss.

## Fit and evaluation

Warm-start EXACTLY the128-step LoRA AND head; keep base weights frozen. No384
weights are used. Rebuild AdamW moments with8-step warmup because targets/mix
changed; LoRA LR1e-6, head5e-6, decay.001, clip1. This is weight continuation,
not a claim of an unchanged optimizer trajectory. At most192 NEW updates.
Validate every16, observe64, then three stale checks (earliest112). Nonfinite
training aborts and severe old expert CE>parent+.50 stops separately.

Score saved, normally loaded checkpoints at EVERY validation. Preserve training
RNG around temporary inference model loading; do not select by an unmerged
training peak that may disappear in normal serving. Record old51 fourth menus
and new6 fourth monitor menus separately. Research primary is an equal-weight
mean of those cohort means; require improvement over the same128-step baseline,
new fourth no drop, old fourth no material drop>1 percentage point, third
terminal utility no drop and holding drop≤.02 (new single-menu guard≤1/32).
All KL distances and fourth escape remain reporting-only. Soft metric progress
extends patience but cannot select weights or prove real match strength.

Only then register a small complete-continuation comparison on already consumed
development games. Fresh independent/full-start pools and formal model/main/EXE
are not changed automatically. Publish code, this plan and aggregate findings;
raw sources/labels/features/research weights stay private.

All source generation, labels, hashes, tests, training and inference run only in
Slurm GPU allocations with companion CPUs, time limits, logs, exit receipts,
managed/reaped children and invalid-dependency cancellation. Login nodes only
connect, manage files and submit/query scheduler jobs. Missing coverage/labels
or provenance failure prevents downstream fitting.
