# Separate the fixed development curve from the endpoint grid

The registered 3B course uses two logical seeds at intermediate checkpoints and
eight at its 2500-update endpoint. Both cross two opponent profiles and four
seats, giving 16 and 64 complete East matches respectively. Plotting the full
64-match result directly on the 16-match curve would change its denominator and
seed population.

`tools/summarize_curriculum_curve16_v2.py` retains the original full journal
points and checks all native records, parameter identities and score ledgers.
Only after all 64 endpoint matches complete does it extract the already
registered first two logical seeds for the comparable 16-match curve. It checks
that aggregate against the evaluator's saved `curve_grid_subset`; it does not
choose a subset by score. The full 64-match result is reported in its own table
and CSV. It remains consumed development evidence.

After all requested points have actually completed, run within a finite
scheduler GPU allocation:

```bash
.venv/bin/python tools/summarize_curriculum_curve16_v2.py \
  --registered runs/curriculum-yaku-early-v2/fit-plan/registered.json \
  --steps 0 625 1250 1875 2500 \
  --out runs/curriculum-early3B2500-summary-v2
```

The output must be a new private directory. `aggregate-comparison.csv` and
`development-summary.md` contain aggregate publication inputs. If the endpoint
is present, `endpoint64-aggregate.csv` is another aggregate publication input.
Original and projected journal points, native records and evidence receipts
remain private. No new games, predictions or parameter updates are generated.

Validation already completed in an allocated GPU job:

- Eight focused arithmetic-fixture tests passed: fixed-seed projection, separate
  full-grid totals, incomplete and duplicate cells, mismatched saved subset,
  wrong grid or parameter identity, intermediate16 handling and reserved-seed
  rejection. These fixtures are not simulated outcomes.
- Reusing the actual completed 0/625/1250 artifacts passed native-ledger checks;
  the aggregate CSV was byte-for-byte identical to the previous report.

The actual 2500-update outcomes are still pending. These checks prove the
reporting behavior, not completion of training or final comparison with v8.
The course, checkpoint selection, production model and unopened confirmation
grid remain unchanged.
