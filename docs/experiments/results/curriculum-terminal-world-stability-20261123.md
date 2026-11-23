# More-world terminal teacher diagnostic: selective declines, incomplete reliability

2026-11-23, Asia/Dubai. The frozen public paired-world planner finishes the
registered TRAIN-only16-view/eight-family diagnostic with1,024 complete
conditional terminal arms. It now produces both useful-looking pursuit
declines and leading accepts, unlike the earlier four-view all-accept pilot.
These are **planner preferences under a sampled prior**, not student behavior,
expert labels, policy win rates or a v8-strength improvement.

## Fixed population and actual work

The [prospective recipe](../plans/curriculum-terminal-world-stability-20261123.md)
selects two distinct TRAIN families in each third_gap3000/fourth_gap5000 ×
ron/tsumo bucket, using audited public small-win proxies and first eligible
IDs. Families are distinct across buckets and have at most one opponent RI.
Each pursuit view has a same-physical hold_second control. No view is selected
by simulated future value or a desired accept/decline label. No validation or
final seed is used in the selection or kernel adjustment.

Each view uses two separately seeded16-world blocks,32 paired worlds total.
Current accept/decline arms share allocations/future seeds within each pair.
Ron decline is native PASS; tsumo decline is one frozen-v8 guarded discard,
not every possible discard. The unchanged utility is top2 + .02*(4-rank)/3
+ .005*tanh(net_from_anchor/8000). Subsequent inner wins are still rule-forced.
Public score views share actual tile histories; they are legal counterfactual
score contexts, not new expert trajectories generated under those scores.

GPU317775 on ws-l3-007 completes0:0 in9m51s; measured model-load/diagnostic
work is582.93s. All16 views and1,024 registered terminal arms finish. Frozen
v8 identity and all224 active-fit dependencies remain unchanged. The failed
first attempt317769 on ws-l2-011 exits at the GPU-resource check after2s:
another compute process occupies the assigned device. The planner never
starts. Preserve its failure; the unchanged entry/helper is retried with that
node excluded, without killing the unrelated process or disabling the check.
Both actual scheduler terminals and shell exits are captured.

## Results include disagreements and secondary-only choices

| Diagnostic | Pursuit views | Same-physical leading controls |
|---|---:|---:|
| Views / underlying families |8 /8|8 /the same8|
| Pooled32-world decline |6|1|
| Pooled32-world accept |2|7|
| Decline with positive sampled top-two difference |5|0|
| Decline with sampled top-two and rank tied, net-only |1|1|
| Two independent block choices agree |7/8|7/8|
| Every leave-one-world choice agrees with pooled choice |6/8|6/8|

The public score proxy does not predetermine a decline: two pursuit views
still favor accepting. Conversely, leading controls are not automatically
correct-accept labels: one favors decline solely through the net-point
tiebreak when sampled top-two/rank tie. The latter agrees across blocks and
leave-one-world checks, illustrating that numerical stability alone does not
make a preference match the intended practical strategy. It warrants explicit
objective/continuation sensitivity rather than silently relabelling it.

The five sampled-top-two pursuit declines provide a more relevant signal than
the earlier synthetic net-only fixture, but they remain conditional on the
uncalibrated world prior and frozen-v8 continuation. Eight source families do
not establish population performance. Leave-one-world checks perturb the same
sample; they are not an independent validation set or a formal confidence
certificate. No positive-CI gate or last-place-rate veto chooses targets here.
The new diagnostics do not justify training every proxy as decline, dropping
all uncertain fourth-place cases, or changing reward from one example.

## Next controlled comparison

The [matched-world continuation recipe](../plans/curriculum-terminal-continuation-sensitivity-20261123.md)
is separately frozen before execution. It keeps all16 views and fixes world
indices0/1 in both saved blocks, four outer worlds per view. First reproduce
all128 cached baseline terminal event streams/ranks/scores exactly; only then
compare the same128 initial arms with bounded future win replanning. The
initial action is fixed. Future planning is depth one, four worlds and at
most four calls per outer arm, with ordinary-win budget fallback and explicit
unsupported-current-kan counts.

Compare ordinary versus replanned values on that **same four-world subset**.
Do not compare four changed worlds to the original32 ordinary worlds and
attribute sampling noise to policy changes. The full32 choices remain context
only. The experiment diagnoses continuation sensitivity; four-world outcomes
will not certify a full32-world adaptive teacher. Its planned inner budget is
at most4,096 additional terminal arms. No target/student admission follows
automatically, and production is unchanged.

The prospective source is214d693942129413dcfab4198217ef1970b1904d, with all four
source/config/entry/plan files exactly read back from GitHub. The32-world
diagnostic source isf860aa9ee981c1bde487bc5450a9753b7270a3f8, also with four
exact source readbacks. Successful/failed bound artifacts remain immutable.
The same calls fit316781 continues toward its registered388 updates; its
final audit and all later-course student/evaluation requirements remain open.

Only code/plans and this aggregate report are published. Source IDs, sampled
hands/walls, per-world outcomes, probability/choice details, complete events
and research weights stay private. All work uses finite scheduler GPU jobs
with companion CPUs, resource checks, timeouts, logs and exits; no login
compute, CPU-only workaround or unmanaged worker is used. No stage4 teacher,
new training label or student update has been admitted, and the final reserved
comparison is unopened.
