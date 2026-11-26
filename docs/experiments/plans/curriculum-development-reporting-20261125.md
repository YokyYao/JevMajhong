# Reporting completed curriculum development checkpoints

The running 3B course records full validation and a fixed 16-match development
grid at steps 0, 625, 1250, 1875 and 2500. The same two logical seeds, two
opponent profiles and four seats allow descriptive comparisons across updates.
The separate 64-match endpoint grid must not be mixed into that curve.

`tools/summarize_curriculum_curve16_v1.py` reads completed checkpoints and
outcomes; it does not generate games or load a model. It reports validation
objective, weighted and natural CE, the new-route loss contribution, observed
reward, hand win rate, deal-in rate, placement rates and net points. Both changes
from the initial checkpoint and from the preceding reported point are retained
in a tidy CSV suitable for plotting.

Run it inside a finite scheduler GPU allocation, after every requested point's
full validation and all 16 native matches have completed:

```bash
.venv/bin/python tools/summarize_curriculum_curve16_v1.py \
  --registered runs/curriculum-yaku-early-v2/fit-plan/registered.json \
  --steps 0 625 1250 \
  --out runs/curriculum-early3B1250-summary-v1
```

The destination must be new. The reporter checks the unchanged execution policy,
validation-data identity, evaluated parameter files, journal values and native
score ledgers. It stores private evidence receipts separately from the aggregate
CSV and Markdown summary. Only the aggregates are publication inputs.

An actual GPU-allocation run on the already completed step 0 and 625 artifacts
passed with exit code 0. It reproduced the previously reported aggregates and
reconstructed their native score ledgers without generating additional games.
The 1250-point comparison remains pending until its actual outcomes complete.

These two consumed seed groups describe development progress; they cannot prove
independent strength, mastery of rare yaku, or parity with frozen v8. The report
does not select a checkpoint, alter training, open reserved final seeds or change
the production model.
