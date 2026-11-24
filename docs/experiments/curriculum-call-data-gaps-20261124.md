# Call-action gaps in the current 3A development evidence

This is a data and cached-prediction diagnostic, not a new training result or
an admission decision. The current paired-order 3A fit remains frozen through
its registered final 2,400 updates. The prepared 3B corpus is unchanged and
has not yet trained a student. No new model forwards, rollouts or optimizer
updates were needed for this diagnostic.

## Existing conditional-simulation targets

The complete fixed 3A validation file contains 23,688 decisions, including
1,184 masked conditional-simulation targets from 140 source families. For
each saved evaluation at 0/600/1,200/1,800, the diagnostic reads the complete
cached prediction file and applies the actual full-legal serving selection
rule, including the existing riichi-class rule. It reconstructs each phase's
original mean selected target mass to within 1e-12.

For the table below, a window qualifies when the largest individual tested
action target for the operation is strictly larger than that of every tested
action of another type. Ties are excluded. Untested actions are not negative
labels. There were no legal-win windows to exclude from this analysis.

| Operation | Qualifying windows / source families | Chosen at 0 | 600 | 1,200 | 1,800 |
|---|---:|---:|---:|---:|---:|
| Chi / 吃 | 113 / 35 | 13 | 19 | 22 | 22 |
| Pon / 碰 | 94 / 41 | 39 | 43 | 50 | 49 |
| Ankan / 暗杠 | 25 / 7 | 4 | 4 | 5 | 4 |
| Daiminkan / 大明杠 | 12 / 10 | 0 | 0 | 1 | 1 |
| Kakan / 加杠 | 6 / 5 | 2 | 2 | 2 | 2 |

At 1,800, the model skips 11 of the 12 qualifying daiminkan windows. Thus
the weak daiminkan result also occurs against the existing simulated soft
targets; it is not only a disagreement with observed human hard labels.
Some chi/pon target alignment improves, while the kan gaps remain.

A target's largest mass is not proof of positive monetary advantage, expert
optimality, statistical confidence or strength. Related windows are not
independent samples. These small, already-consumed development subsets must
not replace complete-game outcomes or the unopened final v8 comparison.

## What the prepared new 3B data adds

These counts concern only the 3,652 new TRAIN and 828 new validation hard
labels, not the inherited replay. A label records the actual Mortal action;
it is not a requirement that the student take every offered operation.

| Offered operation | TRAIN: teacher chooses / other action | Validation: chooses / other action |
|---|---:|---:|
| Chi | 54 / 285 | 10 / 66 |
| Pon | 83 / 147 | 22 / 38 |
| Ankan | 13 / 35 | 1 / 21 |
| Daiminkan | **0 / 17** | **0 / 6** |
| Kakan | 4 / 5 | 1 / 3 |
| Riichi | 148 / 315 | 28 / 35 |

The 17 new TRAIN daiminkan opportunities belong to 17 families; the six
validation opportunities belong to five families. All receive a different
teacher action. This does not establish that these skips are mistakes, but
the new yaku data supplies no additional positive daiminkan examples. The
20% prospective loss allocation to new yaku/control sources therefore must
not be described as a targeted daiminkan repair. Inherited replay still
contains positive daiminkan labels; the whole 3B corpus is not positive-free.

The complete prepared 3B corpus and unchanged numerical recipe are described
in the [dataset card](curriculum-3B-dataset-card-20261124.md). New-source rows
retain their original labels, source tags, legal menus and family partitions.

## Consequences for the next data revision

Continue the registered 3A experiment to its final validation and native
evaluation. Do not edit its inputs, choose an intermediate checkpoint or
relax admission rules in response to this diagnostic. The conditional 3B
entry still requires the originally qualified final 3A parent.

A subsequent call-data repair should collect a prospective, separate set of
ordinary and open-hand opportunities covering all three kan types alongside
chi/pon, including both plausible takes and plausible skips. Freeze whole
source families and TRAIN/validation assignment before labeling. Use paired
continuations from the same public belief worlds for the available actions,
retain uncertain results and quarantine failures explicitly. Do not search
validation outcomes for examples to move into TRAIN, force a kan answer, or
discard valid negative/uncertain cases to meet a desired positive count.

Report operation-level offered/selected counts and family counts before
choosing loss masses. If the frozen collection still supplies too few useful
contrasts, preregister additional TRAIN families separately rather than
resampling held-out families. Keep ordinary/riichi retention and actual
complete-East outcomes in the comparison: improving a kan agreement column
alone does not demonstrate a better player. Any changed data, objective,
mix or candidate lineage needs a new registered experiment identity.

## Evidence identities

The cached diagnostic binds the exact registered validation data, all four
full prediction files, the actual selection-rule sources and the frozen
1,800-step validation journal. It produces no new predictions or games.
The 3B operation count verifies the completed prepared-data SHA-256 values
before and after reading. Aggregate evidence SHA-256 identities are:

```text
3A fixed validation data
bb0fc5ac2820f19f0a33874b8ebe659e59e1accf929c2aa58e3e2570bad3d214

3A cached-call diagnostic private summary
06cad88962361e9846033c305fea8c25bfc0279cdda466038df9d3d0a916a86e

3B prepared combined TRAIN
12a3cef6487c7cdc8a4accab5c4c0bd1e3a5a034dbec82389a2308cfe4f62924

3B prepared combined validation
c77428836780e5b6d68f265f704d39df5f0c7733a13890b75e9e6a7073fa7116
```

Raw hands, teacher labels, world payloads, individual predictions, local
paths and operational logs remain private. This aggregate report changes
neither the running model nor the published Windows model.
