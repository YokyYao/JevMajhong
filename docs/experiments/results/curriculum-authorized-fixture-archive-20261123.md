# Authorized historical-fixture archive and restored regression checks

2026-11-23. The owner explicitly authorizes moving the two failed v1/v2 test
files unchanged to `tests/archive/`, preserving original source/history/failure
records and letting corrected v3 remain in current core discovery. This resolves
the earlier automatic-review rejection; it is not an unapproved deletion or
an attempt to make the failed implementations pass.

Commit a61c739d00ae9c60f93ad241cd0c6706c5008cd4 moves exactly those two files and
adds the [archive README](../../../tests/archive/README.md). Each archived file
reuses the exact original Git blob: v1 a4fdff0179cf012f24f06a365e3c54fd222adc86,
v2 cf68369f323bc802d1ace2c3815d661114bc27ad. Both remain1,005 bytes and have
exact remote text readbacks. Original core paths are absent, corrected v3 is
still present, and no workflow, fixture helper or historical plan is changed.
Old commits/CI failures remain failures; the README gives an explicit GPU-only
command for reproducing archived failures.

GitHub workflow45134761910 on that exact commit completes successfully:

| Actual CI job | Result |
|---|---|
|core137558279989|355 tests, all pass; includes all three corrected v3 checks|
|desktop137558279515|65 tests run, success with3 pre-existing skips|

The source/temporary-fixture/riichi-recovery/post-pon requirements remain
covered by v3. No skip decorator, expected-failure annotation or workflow
exclusion is added to obtain these results. Default discovery remains
`python -m unittest discover -s tests/core -v`.

Separately, bounded GPU317843 on ws-l1-002 completes0:0 in8s. It hashes the
archived files against the original Git blob identities, confirms both old
paths are outside current core discovery, and passes exactly the three v3
native checks with the pinned project/native runtime. All224 active-fit
dependencies remain unchanged. Actual scheduler/shell exits are captured.
The local focused GPU check is not substituted for the full remote CI run.

The current calls training, its bound recipes and completed positive/negative
experiments remain intact. No student update, deployment, Windows rebuild or
main merge follows from a green regression check. All local test/hash work
uses finite scheduler GPU resources with companion CPUs, device/node checks,
logs, timeout and exit records; no login compute is used. Only this aggregate
record, archive source and README are public; detailed run bindings remain
private. The full curriculum/v8 objective is still incomplete.
