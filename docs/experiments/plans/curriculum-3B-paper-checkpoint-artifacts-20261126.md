# Reproduce the Stage 3B paper and usable checkpoint artifacts

The paper distinguishes the pinned-base curriculum from the historical released
policy comparison. The selected ancestry is basic/riichi 608, calls 388, selected
3A 1800, then the registered 3B 2500 budget. The latter is not selected by minimum
validation loss. Frozen v8 is a reference policy, not an initialization model.

## Real learning curves and data mix

`tools/build_curriculum_3b_paper_v1.py` reads a completed source-bound development
report and the actual selected parameter ancestry. It exports LaTeX tables and
PDF/PNG/SVG figures:

- Every actual training objective observation and its EMA (alpha 0.1).
- Fixed-set validation objective on its own labeled zoomed axis, plus validation
  change relative to step 0. Validation values are not smoothed.
- Reward, hand-win rate, top-two rate and net points on the same 16 development
  matches throughout. Lines connect actual recorded observations only.
- Phase-local loss panels for basic/riichi, calls, 3A and 3B. Different phase
  objectives are not joined into a single comparable loss series.
- All 25 source counts and objective masses, checked against the actual frozen
  TRAIN/VAL coverage and scalar-weight formula. There are91,247 TRAIN and24,516
  VAL menus; 19 replay sources receive 80% objective mass and six new sources 20%.

The same 16-match grid has only two consumed logical seeds. Its apparent improvements
are not independent strength evidence. The final 64-match development endpoint
is a separate table; its preregistered first-two-seed subset supplies the
comparable curve. An unchanged observation stays unchanged. Axis ranges
and relative changes help expose small real changes without fabricating a trend.

Actual execution through 1875 completed: figures were inspected, the updated
existing paper compiled twice, and its five new pages were rendered and reviewed.
The paper's 1875 snapshot is explicitly interim, not a completed 2500 result.

## Usable inference export

`tools/export_curriculum_3b_inference_v1.py` copies the exact adapter, head and
tokenizer bytes, includes frozen `jevmahjong`/`jevany` inference source, records
the base revision and library versions, and supplies a request CLI. The pinned
Qwen base weights remain a separate dependency. All execution uses a scheduler
GPU allocation; no login-node inference is permitted.

The packaged policy uses the actual complete-legal-menu RI0.5 rule, with the
existing one-action and available-win shortcuts. It must not be silently replaced
by the historical discard guard or plain argmax. The original training checkpoint,
including optimizer state, remains preserved separately for continuation.

The 1875 export was actually loaded from its copied runtime and answered a
non-shortcut, non-reserved validation request successfully. A 2500 export additionally
requires the genuine completed budget, full saved-weight reload audit, and actual
successful fit/audit/capture terminals. This preparation does not claim those
future prerequisites have already completed.

`tools/attach_curriculum_3b_checkpoint_card_v1.py` creates a new paper-material
version only after the matching 2500 export has passed its load/query check. It
adds the exact parameter identity and inference-policy details without modifying
the original evidence-backed material.

Example commands, inside an allocated GPU job after actual completion:

```bash
.venv/bin/python tools/export_curriculum_3b_inference_v1.py --step 2500 --out runs/your-new-3B-inference-export
.venv/bin/python tools/build_curriculum_3b_paper_v1.py --comparison runs/curriculum-early3B2500-summary-v2 --last-step 2500 --out runs/your-new-3B-paper-material
.venv/bin/python tools/attach_curriculum_3b_checkpoint_card_v1.py --material runs/your-new-3B-paper-material --checkpoint runs/your-new-3B-inference-export --out runs/your-new-paper-with-checkpoint-card
```

Document synchronization uses the user's existing private connection and is
tracked separately. Only paper sources, aggregate tables and figures are sent
to that document project. Experimental weights, private example requests, raw
hands, worlds and process receipts remain private. Packaging and paper updates
do not replace the production model, complete Stage 4 or establish parity with v8.
