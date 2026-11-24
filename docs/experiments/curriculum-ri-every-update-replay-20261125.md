# Prospective replay arrangement: every update sees four RI strata

A candidate TRAIN exposure schedule now preserves every original decision's
total empirical loss weight while ensuring each optimizer update contains all
four inherited riichi accept/decline strata. It is prepared data scheduling,
not a trained model or evidence of higher win rate. The live retained-parent
2,400-update trial continues unchanged.

## Actual complete-corpus preparation

Both candidate epochs were reconstructed from the live trial's registered
baseline permutations, with exact baseline order hashes checked. The original
87,595 unique TRAIN decisions and 23,688 validation decisions are unchanged.
No held-out family enters TRAIN and no label or soft target changes.

| Quantity | Current baseline | Candidate arrangement |
|---|---:|---:|
| Unique TRAIN decisions |87,595|87,595|
| Physical exposures per epoch |87,595|90,228|
| Physical exposures over two epochs |175,190|180,456|
| Optimizer updates over two epochs |2,400|2,472|
| Records per full update |73|73|
| Four basic RI strata present in every update |No|Yes|

The candidate adds 2,633 exposures per epoch, about 3.01%. Every full update has
one example from each of the four RI strata and 69 non-RI examples. Non-RI
original relative order is preserved, with only 42 inserted repeats per epoch
distributed through that order. RI strata cycle through deterministic shuffled
passes. The maximum per-epoch repetition is seven; the largest example weight
is 9.095565, within the existing repeat-eight and weight-sixteen caps.

Repeated exposure does not create new independent data or source families.
This proposal addresses update coverage and sampling variation; it does not
replace the separate targeted-data collection or prove sufficient diversity.

## Original empirical objective is retained

For an original row i in source s, with original source count N_s and actual
repetition c_i, every appearance receives weight
`E * mass_s / (N_s * c_i)`, where E is the epoch exposure count. Its summed
normalized weight is therefore exactly `mass_s / N_s`, independent of repetition.
The maximum observed per-original-row mass error across the complete prepared
epochs is 2.710506e-20. All 19 original source masses are preserved.

This is an identity for the weighted empirical objective at fixed parameters
and identical per-row loss functions; it also preserves the corresponding full
gradient. It does not establish equal stochastic-dropout realizations, Adam
moments or optimization trajectories. Those change when examples are rearranged.

## First-epoch update coverage and effective-mass variation

“Effective mass” here is the sum of the example weights belonging to a source
in an update, divided by that update's record count. The standard deviations
below measure scheduled weight allocation, not measured gradient variance.

| RI stratum | Baseline updates without this source / 1,200 | Candidate updates without it / 1,236 | Baseline mass SD, percentage points | Candidate mass SD, percentage points |
|---|---:|---:|---:|---:|
| Legacy decline |940|0|4.1834|0.2390|
| Legacy accept |1,008|0|4.9460|0.1600|
| Teacher decline |454|0|3.5523|1.4076|
| Teacher accept |355|0|3.4885|1.0056|

All four scheduled mass standard deviations also decrease in epoch two.
Inverse-repeat correction means each update's numerical class weights can
differ; class presence alone is not a claim of exact per-update mass equality.
Lower scheduled-weight variation does not prove that representation interference
or the observed upward reach-preference drift will disappear.

## Validation and next training decision

Four tests pass: complete four-stratum update coverage, deterministic scheduling,
rejection of invalid inputs/caps, and independent empirical loss/gradient
identities under unequal repetitions. Full-corpus preparation checked every
original row's total weight, original baseline orders, both split memberships
and all exposure/repetition limits inside a finite scheduled GPU companion step.
No model was loaded, no student update occurred and no game was generated.

Before training, integrate per-original inverse-repeat weights into a distinct
loss observer/journal and freeze its actual parent checkpoint, optimizer
schedule, evaluation points and budget. Existing source-constant weights cannot
be reused unchanged for this arrangement. Use current-trial development results
and the queued TRAIN-only task-gradient diagnosis to choose the next continuation;
this preparation does not admit a failed parent or select a new model.

Require actual RI retention and native-score results from that future trial.
The present trial, original advancement rules, production v8, main/Windows and
the unopened independent comparison remain unchanged. Only source, this plan's
aggregate results and tests are public; row indices, per-record weights and
raw decisions remain private.
