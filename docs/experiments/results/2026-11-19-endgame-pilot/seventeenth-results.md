# Longer fourth-place head training: learning without a qualified choice gain

The user questioned stopping after two updates. The criticism is justified:
round16's two tiny projected updates and two stale checks did not establish
convergence. A longer, less constrained ablation has now completed128 actual
updates, with all826 positive training rows used on every update. It reached
the registered maximum; it was not stopped after two validations.

## Protocol and verification

Start from formal Play v2/v8 step2088. Same905 train/186 validation membership,
same149 fourth training targets,51 fourth validation windows and fixed source
masses as round16. Existing524,800-parameter head only; encoder and LoRA
frozen. Full-source AdamW LR5e-5, warmup8, validation8, observation32 plus four
later stale checks (earliest ordinary stop64), maximum128. Every-source CE
monotonic projection/line search removed. Fourth escape/native-fourth KL are
reporting-only. Third/holding/replay selection guards retained.

Source95d69a1f, successful CI44515913019; GPU checks309114 exit0 (253core,
89desktop, one optional skip). Fit309115 on ws-l2-019 RTX5000Ada exited0.
Postfit audit sourceec21b8c4/CI44516667777, GPU309120 exit0. All logged
validation probabilities reproduce metrics, selection and stopping state.
The128-step saved LoRA equals formalv8, head differs by relative L2 .1465063
(14.65%) from parent.335 real normal-service queries exactly reproduce cached
probabilities and fourth action choices (maximum difference0, choice flips0).
The lack of qualification is not a model-save/load failure.

## Measured results

| Update | Fourth finite-label top2 at actual greedy choice | Soft fourth top2 | Strict training targets hit (of48) | Original expert validation CE |
|---:|---:|---:|---:|---:|
|0|22.3039%|18.2997%|14|1.00524|
|8|20.4044%|18.1117%|20|1.03241|
|32|21.3235%|18.7220%|18|1.01946|
|64|22.0588%|19.0401%|27|1.08212|
|128|22.0588%|19.1708%|30|1.12048|

These percentages average existing simulated terminal labels over51 fixed
decision menus. They are not complete-game or human win rates. The soft
metric averages model probability over each action's finite top2 estimate;
it is not the top2 outcome of playing the highest-probability action.

No nonzero checkpoint exceeded the formal input-path zero's hard22.3039%.
Consequently selected_step=0 even if all reference KL guards were ignored.
Fourth tail improved59.7426%→57.2917%; it did not veto anything. Some nonzero
points also failed replay/native-third retention guards. At128, third terminal
utility .447962→.449864 and holding .969792→.970833 remain intact, while old
expert CE1.12048 exceeds the formal+0.10 selection allowance. Reference and
native-third KLs also exceed their registered limits. No new complete
continuations or independent sources are spent on this unqualified fit.

## What this establishes

1. The two-update trial was too short to rule out additional learning. With
   the longer trajectory, strict training target hits14→30 and strict-source
   training CE3.99246→.362772. Soft validation also improved after an early
   dip. It would have been wrong to claim the model could not learn more.
2. Continuing the same head is insufficient evidence of better actual
   decisions: none of16 nonzero validations improved hard top2. Training
   success and soft validation movement did not become a qualified choice
   gain. Target coverage/noise and representation limits remain hypotheses.
3. Conflict is measured, not merely assumed. At initialization the ordinary
   strict-fourth gradient has cosine−.776 with original replay and−.826 with
   leading replay. Its norm16.39 versus replay.512/leading.395 means equal
   nominal source weights do not imply equal influence. The combined gradient
   predicts replay loss increases locally. This explains why requiring all
   source CEs to decrease can severely restrict the head direction, and why
   removing that restriction can forget old preferences.

The ablation changed optimizer and observation window together; it does not
isolate either alone. It also does not prove that targets are wrong or LoRA
will solve the problem. The next registered balanced existing-LoRA trial
changes representation capacity, lowers head LR fivefold and ensures every
microbatch accumulation includes all eight sources. It retains the same loss
masses and thus does not claim that balancing alone eliminates conflicts.

All computation ran in bounded Slurm GPU allocations. Formal weights, main
and Windows0.4.2 remain unchanged. Only code/plans/aggregate results are public;
raw traces, features and research checkpoint weights remain private.
