# Stage3 attack/risk: fixed-final candidate recipe

2026-11-21, Asia/Dubai. This recipe is implemented and prechecked while the registered calls course is still running. GPU fit312276 is queued after successful calls audit312082; attack audit312277 depends on fit312276. Invalid dependencies automatically cancel. **Queued is not started: no attack parent is sealed, no attack fit is registered, and no attack optimizer has started.** The launch first requires successful calls-final324 training, its full audit and its original quality targets. The [active four-course goal](curriculum-optimization-20261121.md), stage4 and reserved unseen v8 comparison remain required.

## Parent and data

Continue the same qualified calls student's rank16 LoRA and256-dimensional pointer head on the pinned Qwen3.5-0.8B base. Keep base parameters frozen and start a new optimizer; do not stack a corrective network or resume the parent's optimizer. The main route's cumulative offset would be584+324=908. A local362-step attack fit would end at cumulative1270; the separately trained context-v3 control is not part of that total.

The [prepared informative pool](../results/curriculum-attack-course-preparation-20261121.md) retains22,959 TRAIN/5,905 validation decisions and all20,237/4,721 calls-course replay requests and labels. Whole-group ancestral/held-out/reserved overlap is zero in the actual precheck. Single-support masked targets remain quarantined, not duplicated or hard relabelled.

Keep75% expert loss mass, preserving the calls-course relative plain25%, riichi10%, open25%, call25%, kan15% proportions. The natural194:272 riichi accept/decline ratio and equal per-example weight remain. Allocate25% MC equally across closed discard, open discard, call and kan. Both the training loss and validation arithmetic restore the canonical masked source before computation: an unobserved legal alternative is unlabelled, not bad.

The [source audit](../results/curriculum-stage3-provenance-20261121.md) reproduced all2,722 informative TRAIN MC targets. Their reward is own-win +0.25-speed −1.5-dealin, with no net-point term. These are16/32-world hand-outcome labels with old continuation policies and fixed opponent concealed hands. They are conditional simulation preferences, not public-belief optimality, new self-play or terminal-placement labels. No new confidence filter or reward revision is introduced by this fit.

## Fixed exposure and candidate

The [configuration](../../../configs/training/curriculum-stage3-attack-v1.json) proposes:

| Setting | Value |
|---|---|
| Complete epochs |2|
| Microbatch / accumulation |1 /127|
| Updates per epoch / final |181 /362|
| Actual record exposures |45,918|
| Last accumulation group |99 records,77.95% of a full group|
| LoRA / head peak learning rate |1e-5 /5e-5|
| Validation steps |0,181,362|
| Native development games |16 at0;16 at181;64 at362|
| Candidate |sole registered final362|

Every update must contain all10 expert/MC strata, and every row appears exactly once per epoch. Actual logs must reconcile with private epoch permutations and source counts. Numerical failure stops the bounded job. No early stopping on imitation loss, selection of an intermediate native peak, or automatic epoch extension is allowed. The historical base trainer may still report its minimum-CE result; that result is explicitly not the course candidate.

The native evaluator keeps the same early-course forced-legal-win guard and public inputs as the calls parent. This stage does not teach declining a legal small win. Its step0 tensors must equal the sealed parent, and its16 native games must reproduce the corresponding parent subset before a real parameter update. A discrepancy fails the fit rather than being hidden as a reward change.

## Metrics and development retention

Save actual per-update objective, raw per-source CE/counts, cumulative record exposure, learning rates and preclip gradient norm. SFT training reward stays null. The game reward curve uses observed terminal raw net points/8000, with hand-win, first, top-two, fourth and dealin metrics from complete East matches.

The validation callback reports weighted masked CE, target entropy and CE-minus-entropy KL, plus per-source/phase values. Entropic soft targets have a nonzero minimum CE, so a plateau alone does not mean no learning. Keep expert action agreement separate from MC target-argmax agreement; the latter is not expert accuracy. Explicitly count MC decisions choosing an unlabelled legal alternative instead of treating those alternatives as supervised negatives. Record riichi TP/FN/TN/FP and expert choosing/declining counts for chi, pon and all three kan types.

The final candidate's development aims, relative to its actual step0, are:

- Ordinary exact agreement drop at most2 percentage points.
- Riichi balanced operation agreement drop at most5 points.
- Mean five-call balanced agreement drop at most3 points.
- Equal-phase MC mean CE improvement at least0.02.
- Final64 net-point gain over parent at least0, with no increase in mean dealin rate.

Report full hand-win/first/top-two/fourth results and whole-seed paired95% intervals against both parent and frozen v8. These are development targets, not equivalence or independent strength proof. An imitation gain cannot hide a failed game/risk target. A failed completed recipe remains a documented result and needs a diagnosed new experiment, not silent promotion or changed thresholds.

The [pre-fit daiminkan investigation](../results/curriculum-calls-data-diagnostic-20261121.md) confirms341 selected/321 other-action expert TRAIN examples plus109 labelled MC contrasts. Frozen v8 correctly declines67/108 observed negatives versus the student's3/108 at162. All22,959/5,905 records pass real tokenizer admission. Keep the existing mixture for its planned risk-training test; no blind duplication, forced labels or active-fit change follows. Final audit output now exposes every operation's separate acceptance/decline rates and can recheck an explicitly bound cached v8 menu diagnostic. Aggregate call agreement alone does not establish that this failure has disappeared. Budget, data, losses and quality thresholds remain unchanged.

Use the existing development grid:8 logical seeds2026200000–2026200007 ×4 seats ×2 opponent profiles. All three opponents are EfficiencyAgent in the ordinary profile. The calling profile replaces the next-seat opponent with CallingOpponent. Cache parent64 and v864 only after identity/grid checks; do not recompute them at every point. New student games total96, and the complete audit reconciles224 games including references.

Curve points always use the same first-two-seed16-match subset, including the final64 subset. Keep the separate full64 mean in its own CSV/table. Do not join different objective mixtures into a single smooth loss line, label synthetic arithmetic probes as model observations, or use the reserved final seeds2026300000–2026300031 for tuning. PNG/PDF plus aggregate CSV/provenance exports remain required.

## Execution after a real parent qualifies

All commands below run inside a bounded scheduler GPU allocation, never on login. Use fresh output paths. First seal the actual completed calls final, preserving successful fit/audit exits and captured scheduler completion snapshots:

```bash
python tools/curriculum_calls_parent.py \
  --registered runs/curriculum-stage2-calls-v1/fit-plan/registered.json \
  --evidence runs/curriculum-stage2-calls-v1/final-evidence-private.json \
  --fit-exit "$CALLS_FIT_EXIT" --audit-exit "$CALLS_AUDIT_EXIT" \
  --observations runs/curriculum-stage2-calls-v1/scheduler-observations-private.jsonl \
  --child-data-ready runs/curriculum-stage3-attack-v1/data/data-ready.json \
  --out "$CALLS_FINAL_SEAL"
python tools/register_curriculum_attack.py \
  --config configs/training/curriculum-stage3-attack-v1.json \
  --parent-seal "$CALLS_FINAL_SEAL" \
  --operation-diagnostic runs/curriculum-stage3-attack-v1/calls-data-audit-v1/summary-private.json
python tools/train_curriculum_attack.py \
  --registered runs/curriculum-stage3-attack-v1/fit-plan/registered.json
python tools/audit_curriculum_attack.py \
  --registered runs/curriculum-stage3-attack-v1/fit-plan/registered.json \
  --binding "$ATTACK_AUDIT_BINDING" \
  --out runs/curriculum-stage3-attack-v1/final-evidence-private.json
```

The registrar binds source, prepared data, parent seal, actual trainer arguments, tests, scripts and cached reference files before fitting. Actual launchers/audit bindings must also be hashed before the first update. `cluster/curriculum-stage3.slurm` provides GPU1/CPU4/48GB with a6-hour limit; its audit template provides45 minutes. Submit the audit afterok the fit, cancelling invalid dependencies. Both produce logs and exit receipts; all foreground subprocesses have time limits. No login computation, CPU-only workaround, unmanaged background work or model publication occurs.

Run `tools/audit_curriculum_calls_data.py` as documented in the investigation before using `--operation-diagnostic`; it generates that host's fixed reference cache. The flag is optional for the minimal reproduction route and its absence stays explicit. In the supervised main route this available cache will be bound. It changes final diagnostic reporting, not gradients, early stopping or candidate selection.

The final audit rechecks exact step0 parent tensors and native subset; all362 actual updates,10-source exposures and optimizer state; every stored validation profile; a reload of all5,905 final menus; complete-game ledgers and paired results; source/input hashes; and figure provenance. Software-evidence pass is separate from stage3 quality pass.

## What has actually passed

The real prepared-data precheck already verifies API replay, ancestry,160 loss/gradient comparisons with zero difference and two complete sampler permutations. The new entry precheck compares portable target CE to the frozen research implementation on all5,905 real validation menus: maximum difference0. Those probability vectors are synthetic arithmetic fixtures, with no model loaded, optimizer updates or curve points.65 curriculum regression checks pass, including5 new checks for masked validation, invalid/dropped predictions, fixed362-step selection, quality regressions and whole-seed pairing.

All71 curriculum regressions and GitHub core/desktop CI44850971429 passed before submission. The calls-parent reader now accepts omitted zero-valued Counter keys from the actual frozen auditor; positive/negative coverage, integer validity and all original quality thresholds remain enforced. This compatibility fix and the launch/audit source bindings were frozen before queueing, without changing active calls training files.

Actual qualified-parent loading, real trainer callbacks/updates, final inference reload and full attack audit still require the completed calls parent. There is no attack performance result yet. All new weights and per-record predictions/source/world/order details stay private. Only source, tests, plans and aggregate results are intended for this research PR; the practical v8 Windows tool remains unchanged.
