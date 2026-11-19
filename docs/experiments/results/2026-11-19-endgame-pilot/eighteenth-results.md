# Balanced joint LoRA: small finite-label gain, not a match-strength result

Round18 continued formal Play v2/v8 step2088's original rank16 LoRA and head
for128 optimizer updates with32-record source-balanced accumulations. All826
positive rows were covered by39 updates. Training and the automated normal
serving audit completed successfully (GPU309132 and309142, both exit0).
Training took about919 seconds including validation. No complete simulations,
new labels, independent pools or deployment were performed.

| Fourth validation comparison | Finite-label top2 at actual chosen action |
|---|---:|
| Formalv8, same endgame input/action path |22.3039%|
| Step96, live unmerged training evaluation |23.0392%|
| Step128, normal loaded serving |22.5490%|

These are existing terminal-label averages over51 decision windows, not
complete-match or human top2/win rates. At round18 completion, step96 had
not separately undergone normal serving; the newly registered duration
extension checks96 and128 before selecting its stronger comparison floor.
That extension preflight (GPU309187) has now loaded both checkpoints and
made335 requests per checkpoint: BOTH yield22.5490% in normal serving.
The live unmerged96-step peak23.0392% does not survive the normal inference
path, so it is not counted as a verified checkpoint gain. The continuation
uses the actual22.5490% parent floor.
The original released-input reference is16.6054%, but comparing that with
the new endgame path confounds input/routing and learned parameter changes.

Step128 normal serving made335 requests (186 validation and149 fourth TRAIN
diagnostics), verified actual changes of both existing LoRA and head, and
reproduced all186 validation greedy decisions from live evaluation. Maximum
probability difference was .0386066; BF16 unmerged and loaded/merged serving
must not be assumed numerically identical. Actual serving metrics, not
probability identity, determine the following result.

The hard fourth objective improved .2230392→.2254902, about+.2451 percentage
points. Third terminal utility stayed .447962; holding .969792→.970833.
Original expert CE1.09656 remained within formal+0.10. The one failing
historical protection was native-third preference KL .145359 versus limit.05.
Thus selected_step=0 under the registered round18 protocol. Fourth escape
did not veto it. This is not proof of actual third-place outcome regression;
the KL measures preference-distribution distance on8 native-third menus.
Conversely, unchanged finite-label third utility is not proof of complete
match safety. No release qualification is claimed.

Late live soft fourth values continued .188227 at80→.189090 at96→.191541
at112→.192140 at128. That supports testing a longer trajectory, while the
hard peak at96 and repeated small validation set also permit overfitting.

The user requested more duration or data. The next experiment preserves all
labels/masses/memberships and resumes final128 chronologically with optimizer
moments/RNG/sampler, through at most total384. New outcome-based research
screening is registered before updates, reports all historical guard decisions,
and requires later complete-continuation evidence before any deployment.
It does not retrospectively qualify round18 under altered rules.

Source5d8a484a and CI44517767271 passed; GPU checks309129 passed256 core and89
desktop tests (one optional skip). Normal-serving audit source5745b154 and
CI44518642858 passed. Earlier sampling test failure309126 was corrected
before model training. Raw traces, features and research weights are private;
only code/plans/aggregate findings are public. Formalv8/main/Windows0.4.2
remain unchanged.
