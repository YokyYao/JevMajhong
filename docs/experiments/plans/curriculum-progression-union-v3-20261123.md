# Union v3: current whole-family holdout files override inherited origin markers

Preparation v1/317986 fails on an absent legacy row split. V2/317999 handles
absence but still incorrectly rejects supplied historical origin markers.
Both source versions, tests, partial outputs and actual failures remain intact.
No student update or original data change results from either preparation.

Read-only GPU318019 on ws-l2-008 diagnoses every source file. Current calls VAL
contains4,401 legacy records whose inherited `split` says TRAIN; current MC VAL
has745 such records and219 without a marker. MC TRAIN also has501 without one.
All Mortal family markers match their current files. These are historical
producer metadata, retained when stricter curriculum whole-family holdouts were
created. They are not permission to move heldout records into training.

V3 keeps every original row object and marker unchanged, using the hash-bound
current source file as partition authority for inherited data. It still requires
explicit correct per-record splits for Mortal source families, and rejects
opposite-split ID/group/public-input overlap and the675 original reserved TRAIN
groups. The new focused check specifically keeps a legacy TRAIN origin marker
inside its bound VAL partition while rejecting a wrongly split Mortal family.
No label, record, group, source file or loss mask is rewritten to pass the check.

All [original union requirements](curriculum-progression-union-20261123.md)
remain87,595/23,688 informative records plus136/41 structural quarantine,
unchanged actual calls replay, exact original conditional MC and source hashes.
Count missing and stale origin markers separately. Recheck bound TRAIN-world
evidence only, no validation rollout payloads/new simulation/model loading.
Use a new helper/config/entry/test and `data-v3` output. Preserve failures and
their original expectations. The future independent audit must use the same
source-file authority and compare exact source row objects, tokens and masks.

All computation stays in a finite scheduler GPU job with device checks, log,
timeouts/exit and224 active-fit dependency checks. This is preparatory data work,
not a new loss mix, parent admission, training launch or model-quality result.
