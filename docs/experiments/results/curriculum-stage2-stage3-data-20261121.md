# Stage2/3 data preparation before fitting

2026-11-21, Asia/Dubai. The fresh-base stable-v2 stage1 fit is active as GPU job311884. The work below prepares later-course data only; no stage2/3 student, optimizer or parent checkpoint was loaded or admitted. It does not demonstrate model improvement.

All audits/builders ran as bounded foreground steps inside the existing allocated GPU job with companion CPUs. The active stage1 input files, weights and frozen configuration were unchanged. New derived records and detailed diagnostics remain private; scripts and this aggregate report are public.

## Expert calls, kan and replay

The stage2 pool contains20,237 TRAIN decisions in2,416 source groups and4,721 validation decisions in739 groups. Every record is an expert-replay decision. All9,292/1,972 stage1 decisions remain with exactly the same model-visible input and observed label. Original/reserved validation groups are excluded from this and the already prepared later-stage training.

| Primary phase | TRAIN | Validation |
|---|---:|---:|
| Closed discard/riichi replay |9,292|1,972|
| Open-hand discard |5,590|1,220|
| Call window |3,569|961|
| Kan window |1,786|568|

Primary phases are exclusive; operation opportunities are not. A kan window can also offer pon/chi or riichi. Not selecting one operation can mean choosing a different call, not just passing. The following counts must therefore not be summed into a dataset size.

| Operation | TRAIN selected / offered | TRAIN positive / negative source groups | Validation selected / offered | Validation positive / negative source groups |
|---|---:|---:|---:|---:|
| Chi |715/2,263|558/922|211/605|161/244|
| Pon |738/2,079|582/1,075|215/586|157/305|
| Ankan |445/895|431/400|136/269|129/119|
| Daiminkan |341/662|325/306|88/196|85/103|
| Kakan |110/231|108/115|57/103|55/42|
| Riichi, including kan windows |199/534|169/251|42/121|36/63|

Chi selections include449 closed-hand and266 open-hand TRAIN cases, plus124/87 in validation. A concealed kan keeps a hand closed; a regression check covers this explicitly. These are useful coverage counts, not proof that the observed call is optimal. Opponent-riichi and turn-context counts are retained in the private coverage receipt for later diagnostics.

The proposed stage2 dataset retains every row once and only changes a loss-only source tag, outside model input. Its proposed loss masses are plain25%, closed riichi10%, open-discard25%, call25%, kan15%. Within the closed riichi category, teacher accept/decline retain the TRAIN ratio194/272. Per-example weights are approximately0.57322 plain,4.34270 either riichi class,0.90505 open,1.41755 call and1.69964 kan, below the cap8. No example is duplicated and no label is changed.

These are data proposals pending an accepted stage1 parent and separately frozen fit budget, learning rates, replay checks and candidate rule. They are not a claim that stage2 training has started. Stage1 retained validation and full-operation choose/decline checks must accompany eventual complete-game comparisons.

## Simulation targets: remove zero-information records, preserve masking

The source stage3 pool contains20,237 expert +2,858 simulation TRAIN decisions, and4,721 expert +1,225 simulation validation decisions. The simulations use `paired_future_masked` targets: zero mass on an unobserved legal action means unlabelled, not known-bad. Future phase weights must restore/preserve that masked-loss identity.

The audit found136 TRAIN and41 validation simulation targets with only one positive-support action, all in open-discard windows. Under masked CE, a one-element log-softmax is identically zero; these examples provide no comparison gradient. The builder quarantines them using support structure alone, without inspecting reward signs or changing a preference label. A gradient test confirms zero loss/gradient; another confirms that even extremely large logits on an unobserved action do not alter the phase-weighted masked loss or gradient.

| Prepared population | TRAIN | Validation |
|---|---:|---:|
| Expert retained unchanged |20,237|4,721|
| Informative simulated comparisons |2,722|1,184|
| Combined fitting/validation data |22,959|5,905|
| Single-support records quarantined separately |136|41|

Informative simulated TRAIN phases are717 closed discard,254 open discard,1,503 call and248 kan. Validation has457/100/477/150 respectively. The full original5,946-row validation file is preserved as a separate frozen legacy reference; the informative validation population has changed and must be labelled separately in curves/reports. No original held-out group can flow back into training merely because its zero-information rows were quarantined.

Of the original simulated targets,1,115 TRAIN and640 validation menus leave some legal actions unlabelled. Among comparisons with at least two support actions,616 TRAIN/243 validation targets have normalized entropy at least0.95, and817/395 have top-two probability gap below0.1. These ambiguous preferences are retained. Entropy or gap describes stored preferences; it does not provide Monte Carlo uncertainty.

None of the published per-record metadata retains world count, independent world outcomes, reward formula or continuation-policy version. This audit therefore does not reconstruct or certify independent-world confidence. New simulation labels need their own public-information generator, frozen reward/world budget/policies and whole-source-group splits. Do not invent missing provenance or filter near-uniform labels as if their optimality were known.

The portable public arithmetic in `tools/curriculum_losses.py` is also checked against the actual frozen research losses, rather than relying on a locally installed research module absent from the public runtime tree. On32 expert +128 informative simulated +136 singleton records,592 plain/phase-weighted loss and gradient comparisons have maximum difference0. The frozen fit's package remains unchanged. This fixes a public-CI import dependency without skipping the gradient checks or changing the ongoing trainer.

A proposed starting loss mix is75% expert/25% simulation, with equal mass across the four simulated primary phases. It yields MC per-example phase weights approximately2.00131 closed,5.64936 open,0.95472 call and5.78604 kan, below cap8. This is a bounded proposal, not a completed fit or evidence that it outperforms another mix. Parent retention and matched complete-game outcomes decide its usefulness.

## Reproduction and identities

After preparing the pinned public source data as described in the [reproduction guide](../../../reproduction/README.md), run inside a scheduled GPU allocation:

```bash
python tools/audit_curriculum_calls.py --data-ready runs/curriculum-reproduction-v1/clean-data-v1/data-ready.json --out runs/curriculum-optimization-v2/stage2-coverage-private.json
python tools/build_curriculum_stage2.py --data-ready runs/curriculum-reproduction-v1/clean-data-v1/data-ready.json --out runs/curriculum-optimization-v2/stage2-data
python tools/audit_curriculum_simulation.py --data-ready runs/curriculum-reproduction-v1/clean-data-v1/data-ready.json --out runs/curriculum-optimization-v2/stage3-target-audit-private.json
python tools/build_curriculum_stage3.py --data-ready runs/curriculum-reproduction-v1/clean-data-v1/data-ready.json --out runs/curriculum-optimization-v2/stage3-data
python tools/check_curriculum_losses.py --registered runs/curriculum-optimization-v2/fit-plan/registered.json --data-ready runs/curriculum-optimization-v2/stage3-data/data-ready.json --out runs/curriculum-optimization-v2/loss-parity-private.json
```

Builders reject an existing output directory. In the supervised workspace the corrected call-coverage receipt is named `stage2-coverage-corrected-private.json`; it supersedes the first receipt's crude existing-meld context labels. Training data and operation totals did not change. Data/label/group hashes and all invariants passed in the actual allocated steps, as did two concealed-kan checks and two masked-gradient checks.

| File | SHA256 |
|---|---|
| Stage2 TRAIN |`81359fee88715b40457c5162d94f5fc84465898422968bfb3683d1aff3bf607b`|
| Stage2 validation |`6d6041d5978423e5870bd03b5f1d1aa238df98eea4ee83268e596cf73abb51c8`|
| Stage3 informative TRAIN |`e0641815f27808b6eb18298c63d5a72b0043d62c6de9254f875b7b15c3f74205`|
| Stage3 informative validation |`8e2be0bec9ef12e4f2df51a0afdcc9483773db0fd9bbec8d817f805540ec5bb6`|

The [full optimization goal](../plans/curriculum-optimization-20261121.md) remains active. Stage1 quality, stage2/3 fitting, a valid win/pass endgame course and the preregistered unseen v8 comparison are still required. Prepared datasets are progress, not completion of those requirements.
