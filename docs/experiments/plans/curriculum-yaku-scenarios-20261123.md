# Stage3B source pilot: native yaku-route initial decisions

The existing catalog finds only2 TRAIN near-kokushi records and1 near-chuuren
base-pattern record. Generate512 new source decisions to begin planned stage3B,
while the fixed stage3A optimizer continues independently. This pilot creates
unlabeled source material, not a stage3B fit or a claim of acquired skill.

Use kokushi, chuuren, suuankou, daisangen and chiitoitsu comparison templates.
For each of16 templates/route, remove1,3 or6 target tiles and replace until a
13-tile starting hand is formed; add two independently sampled physical world
variants. This gives480 targeted decisions plus32 unselected random-initial-hand
controls. First-draw noise is outside the intended complete target tile types.
Damage is a construction descriptor, not actual shanten, difficulty, an optimal
route label or a requirement to keep pursuing the target.

Construct all plans before native collection. Group matching initial13 shapes
and decision14 shapes, including suit permutations and independent suit rank
reflections, into connected families. Assign whole components by deterministic
hash to TRAIN/validation. Do not claim512 independent families or a guaranteed
per-route holdout size. Preserve these assignments for later derived rollouts
and augmentation; do not move cases to fill a quota. Existing final reserved
seeds are not used. Target route tags and complete plans never enter model input.

Every wall contains all136 unique physical tiles. Deal it through the pinned
native Mahjong Soul East/red-five engine, with dealer/observer0 at the initial
turn, and check the actual physical14-tile observation. Reuse the pinned public
request/menu builder, which rejects other concealed hands. Save source wall,
seed, complete native initial prefix, plan/group/split and unlabeled request
privately so later teacher collection can replay the same case. Never add fake
action labels or copy hidden state into requests. Check cross-partition family,
public-request and canonical-shape separation while writing the source files.

Limitations are explicit: first-turn/dealer-only, no observed opponent threat,
no executed abandonment, no completed hands/matches, no teacher labels, no
tokenizer/fit admission and no natural-distribution strength estimate. Later
work must add native midgame contexts and viable alternatives, followed by
public-observation teacher or uncertainty-aware public-belief simulation labels.
The sampled true wall is a replay source, not permission to label actions with
clairvoyant access. Full-game references stay separate from the targeted set.

Use one finite15-minute GPU job with companion CPUs, resource checks and logged
exits; no login computation, CPU-only workaround or detached process. Bound
generator/config/native/request-builder inputs before collection; preserve any
failure under a new version. Raw data stay private. Publish code/plan and actual
aggregate coverage only. Do not alter current stage3A weights, data or mixture.
