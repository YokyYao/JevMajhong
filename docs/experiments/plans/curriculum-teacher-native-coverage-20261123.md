# Independent teacher turn-count and execution audit

The running risk collector's `counts.*.own_discard_number_*` incorrectly counts
characters in the public river string rather than discarded tiles. Its window
selection, source trajectory, targets and guard diagnostics do not use that
counter. Preserve the frozen collector, all rows and its original summary. The
affected aggregate field is invalid and must not be used as turn coverage.

After the complete independent risk audit, run
`tools/audit_curriculum_teacher_native_coverage_v1.py` in a bounded GPU allocation.
Reconstruct every audited basic, advanced and additional risk teacher row from
its actual native source prefix. Compare the entire normalized serving input,
count physical native discards and cross-check the public river's tile tokens.
Publish only the resulting aggregate replacement histogram, with the error
disclosed. Reproduce the old erroneous histogram exactly as a positive check of
the error's origin; do not overwrite it or reinterpret it as valid.

For every reconstructed row, audit the unchanged PolicyAgent eligibility union
and classify teacher-target exclusions by shanten ceiling or existing rule-risk
ceiling, source, split, teacher action and opponent-riichi count. This quantifies
whether a learned teacher action could actually be selected; it does not prove
the teacher optimal, remove the serving guard or admit a risk fit. In particular,
increasing shanten to fold may be excluded even when the discarded tile is safe.

On actual source-selected reach decisions, execute the real declaration in the
isolated native replay and count the next active menu. This checks the declaration
and following discard interface; it creates no new labels or model updates and
does not demonstrate strategic riichi quality. Keep private per-row diagnostics.

All current fit, collector, audit, completed-source and control bindings remain
unchanged. No base model is loaded. A timeout or failure is retained; no partial
audit is described as complete. Subsequent training changes need their own
prospective configuration and full-game comparison. The four-course goal and
unopened final v8 evaluation remain required.
