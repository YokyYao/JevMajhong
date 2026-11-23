# Historical failed native fixtures

The owner explicitly authorized moving these two files here on2026-11-23.
Their bytes and Git blob identities are unchanged; original commits and CI
failure records remain available. This is an archive of incorrect fixture
implementations, not a test skip/expected-failure annotation or a workflow edit.

| Archive file | Original path | Unchanged Git blob |
|---|---|---|
|test_curriculum_observer_declines_v1.py|tests/core/test_curriculum_observer_declines_v1.py|a4fdff0179cf012f24f06a365e3c54fd222adc86|
|test_curriculum_observer_declines_v2.py|tests/core/test_curriculum_observer_declines_v2.py|cf68369f323bc802d1ace2c3815d661114bc27ad|

v1 source92c35bcf11b5bd51b086cdd64de20cf20a7c4262 fails three native checks:
riichi is an untiled action followed by a separate discard, normalized menus
merge equivalent physical copies, and the permanent RI-furiten Python getter
is not exposed. CI45115740594/core137498050262 retains those failures.

v2 source92d58ba6d98fa5a1a4ac6700740752b4ccf8b646 adds three failed checks:
empty native response windows auto-advance, and immediately discarding the
called East after pon violates kuikae. CI45116523669/core137500529703 retains
both versions' six failures. The v1/v2 fixture helpers and historical plans
are unchanged. Neither version completed a successful GPU native audit.

The corrected tests remain in
`tests/core/test_curriculum_observer_declines_v3.py`, backed by actual prior
legal-win declines, native behavioral furiten recovery and a legal post-pon
discard. GPU317302 passed those three checks plus sampled-world and full
remaining-match controls; see
[the result](../../docs/experiments/results/curriculum-observer-declines-20261123.md).
Default core discovery is still `python -m unittest discover -s tests/core -v`.

To reproduce either known historical failure explicitly, use a bounded GPU
allocation with the pinned native runtime and run, for example:

```bash
python -m unittest discover -s tests/archive -p test_curriculum_observer_declines_v1.py -v
```

Failure is expected for those archived implementations. Check out the original
source commit to restore their exact original paths. No training, inference,
tests, hashing or persistent polling may run on a login node.
