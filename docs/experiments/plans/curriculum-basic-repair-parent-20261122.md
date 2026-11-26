# Source-specific admission after the new basic course

The basic-repair fit plan already requires separate aggregate, unchanged-legacy and teacher quality checks. This implementation makes that boundary explicit before the running608-step candidate is available. It does not alter the fit, its frozen grades or the final-v8 objective.

`tools/seal_curriculum_basic_repair.py` first verifies the actual final validation step, all six source counts, separate riichi confusion counts and their aggregate partitions. It recomputes the three quality grades rather than trusting a success header. Missing source results, forged/pool-only success, failed legacy/teacher quality and an unfinished checkpoint are rejected. The original full-budget, initialization/reload, successful scheduler exits, input bindings, held-out separation and complete child replay checks still run. There is no research override in this wrapper.

After a qualified complete final and a separately admitted child dataset exist, use this wrapper rather than the older aggregate-only sealer:

```bash
python tools/seal_curriculum_basic_repair.py \
  --registered runs/curriculum-basic-repair-v1/fit-plan/registered.json \
  --evidence runs/curriculum-basic-repair-v1/final-evidence-private.json \
  --fit-exit "$FIT_EXIT_RECEIPT" --audit-exit "$AUDIT_EXIT_RECEIPT" \
  --child-data-ready "$CHILD_DATA_READY" --out "$SOURCE_QUALIFIED_PARENT_SEAL"
```

Run in a bounded GPU allocation while successful scheduler states remain available; retain actual scheduler terminal receipts immediately when jobs finish. A missing history must not be invented as success. This wrapper has no fallback that rewrites expired job history. Existing research/raw failures remain preserved. Sealing an actual new parent, checking its full loading path and preparing the child replay are still pending the real completed fit/audit/data; synthetic rejection fixtures are not a claim that those operations have succeeded.

The seal binds this source-quality implementation and the grade implementation alongside the original parent artifacts. Later-course registration must use the resulting source-qualified seal and retain every old/new basic input/label/group. No stage2 fit or release is launched by creating the helper. If the final source grades fail, report and diagnose the failure rather than silently using the older aggregate-only tool.
