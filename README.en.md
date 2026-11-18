# Jev Mahjong

[简体中文](README.md) | **English**

A local discard and call recommendation tool for four-player riichi mahjong in Mahjong Soul, with supporting simulation and training code.

[Download the Windows installer](https://github.com/YokyYao/JevMajhong/releases/download/desktop-v0.4.2/JevMahjong-Setup-0.4.2-win-x64.exe) · [Desktop guide](desktop/README.md) · [Model catalog](weights/README.md) · [Training and evaluation](docs/training.md) · [Experiment records](docs/experiments/README.md)

Windows 0.4.2 reuses the installed runtime. An in-place upgrade downloads only changed application and model components while preserving screenshot calibration and the base-model cache. Missing or damaged runtime components are restored automatically. [Upgrade guide](desktop/README.md)

## Current model: Jev Play v2

Windows **0.4.2** uses **Jev Play v2** by default. Versions 0.4.0, 0.4.1, and 0.4.2 use the same model parameters; the latter two releases primarily improve desktop recognition, interaction, installation, and upgrades.

| Item | Details |
| --- | --- |
| Backbone and architecture | Qwen3.5-0.8B with the open JevAny candidate-action decision interface |
| Trainable parameters | One shared rank-16 LoRA adapter and action head; the backbone is frozen |
| Supported decisions | Discards with a closed or open hand, riichi, winning by ron or tsumo, chi, pon, concealed kan, open kan, added kan, and pass |
| Released checkpoint | The final checkpoint at step 2,088 of experimental v8, released as **Jev Play v2** |
| Runtime | Local screenshot recognition on Windows; the model scores candidate actions from structured, publicly visible game state |
| Public resources | [Inference weights](weights/jev-play-v2/README.md), [training and validation data](datasets/jev-play-v2/README.md), and [historical training implementation](https://github.com/YokyYao/JevMajhong/tree/2f0c84a3f9c813990da17ef7dc4ba13f7b68979e) |

### Training

Jev Play v2 combines imitation of expert game records with simulated action comparisons. This round continued training the LoRA adapter and action head from v7 for one full epoch over **33,408 training positions**, with an effective batch size of 16 and **2,088 optimizer updates**. A separate set contains **620 validation positions**, with source groups kept disjoint between training and validation. The 2,088 steps refer to this round, rather than the model's total training history.

| Decision type | Training positions | Validation positions |
| --- | ---: | ---: |
| Closed-hand discard | 15,608 | 332 |
| Open-hand discard | 8,448 | 80 |
| Chi / pon window | 6,728 | 80 |
| Kan window | 2,624 | 128 |
| **Total** | **33,408** | **620** |

The training set contains 29,625 expert examples and 3,783 examples with simulation-derived targets. Expert and simulation examples receive **75% / 25%** of the training loss weight. Within the simulation component, closed-hand discards, chi/pon, kan, and open-hand discards receive **40% / 30% / 20% / 10%**, respectively. These are loss weights, not proportions of the record count.

Simulation targets compare candidate actions from the same position under shared sampled futures. The reward is **own-win indicator + 0.25 × discounted own-win reward − 1.5 × deal-in indicator**, with a speed discount of 0.98 and zero weight on net points. Chi, pon, and kan receive no fixed bonus simply for being selected. Simulation targets are subject to sampling and rollout-policy bias and are not expert ground truth. [Data and weighting](datasets/jev-play-v2/README.md) · [Training plan for this round](docs/experiments/plans/unified-attackrisk-lora-v8-20261115.md)

### Simulation results

The four-player East-only confirmation evaluation used **32 new wall groups, 3 opponent configurations, and all 4 seats**, for 768 matches across two policies, with 384 matches per policy. Opponents included rule-based policies and a historical model. The results describe this simulation protocol.

| Metric | Experimental reference combination | Jev Play v2 |
| --- | ---: | ---: |
| First-place rate per match | 28.65% | **33.59%** |
| Top-two rate | 56.77% | **64.32%** |
| Fourth-place rate | 22.66% | **19.01%** |
| Mean placement | 2.3724 | **2.2109** |
| Mean net points per match | +1,911.72 | **+3,200.26** |
| Hand-win rate, equally weighted by match | 25.26% | **27.75%** |

The reference is the **experimental combination** of Discard v1 + Call v1 + Call Tile Win v1, which differs from the original Jev Play v1 desktop release. The net-point difference is **+1,288.54 points per match**, with a paired 95% interval of **[−298.74, +2,875.83]**. The hand-win-rate difference is **+2.50 percentage points**, with an interval of **[−0.56, +5.55]**. Both intervals include zero, so a consistent improvement has not been established. These results also do not establish superiority over the original desktop release or improved win rates against human players. The 768 matches come from 32 independent wall groups and must not be treated as 768 independent samples. [Full confirmation summary](docs/experiments/results/2026-11-12-play-input-audit/unified-attackrisk-v8-confirmation-summary.json)

### Planned training requirement: endgame placement

Record all four players' scores and placements after each hand. In the endgame, choose comeback or placement-preserving lines using actual settlement, score gaps, and continuation or extension rules. This requires terminal-match placement targets and corresponding changes to the desktop win/pass path. This feature is not yet implemented. [Requirement and training plan](docs/experiments/plans/endgame-placement-20261119.md)

## Repository layout

| Directory | Contents |
| --- | --- |
| `desktop/` | Windows UI, screenshot recognition, and packaging configuration |
| `jevmahjong/` | Reusable game state, policies, data preparation, training, and evaluation |
| `datasets/` | Published training and validation decision samples with verification manifests |
| `weights/` | Model catalog, public weights, and restoration recipes |
| `configs/` | Current training and evaluation templates and environment lock records |
| `tools/`, `cluster/` | Weight restoration, resource preparation, packaging, and general job entry points |
| `docs/experiments/` | Historical plans and results; one-off programs have been removed from the current branch |
| `tests/` | Retained core and desktop regression checks |

## Development

Use Python 3.12. On a server, first install a PyTorch build compatible with its GPU driver, then prepare the project environment:

```bash
python -m pip install -e '.[train]'
python tools/fetch_vendor.py
python -m pip install -e 'vendor/JevAny[train,fast]'
python tools/prepare_models.py
python -m jevmahjong --help
```

Jev Play v2 [training and validation data](datasets/jev-play-v2/README.md) and inference weights are public. Raw per-match logs, run directories, and other experimental weights remain local. Quick regression checks are separate from full match evaluations; see the [development guide](docs/development.md).

## Citation and acknowledgements

This project's decision interface, training integration, and inference integration are based on **[JevAny](https://github.com/SimpleJev/JevAny)** by **Tianxin Wei**, licensed under Apache-2.0. This project pins commit [c740121](https://github.com/SimpleJev/JevAny/tree/c740121a4532592b09b6272104d5f29a94c232e5). We thank JevAny for its open decision-model infrastructure.

Following that version's [CITATION.cff](https://github.com/SimpleJev/JevAny/blob/c740121a4532592b09b6272104d5f29a94c232e5/CITATION.cff), please cite JevAny when using the JevAny-based work in this project, and retain attribution to its upstream sources:

```bibtex
@software{wei2026jevany,
  author  = {Wei, Tianxin},
  title   = {JevAny},
  version = {0.3.0},
  date    = {2026-09-30},
  year    = {2026},
  url     = {https://github.com/SimpleJev/JevAny},
  license = {Apache-2.0}
}
```

JevAny's infrastructure includes modified code from [Kev](https://github.com/jaredpalmer/kev), created by **Jared Palmer** and licensed under Apache-2.0. Its related components also credit [Hugging Face Transformers](https://github.com/huggingface/transformers), [Cygnet](https://github.com/blockbrain-ai/cygnet-recipe), and [NInfer](https://github.com/igorls/ninfer). See the upstream [ACKNOWLEDGEMENTS.md](https://github.com/SimpleJev/JevAny/blob/c740121a4532592b09b6272104d5f29a94c232e5/ACKNOWLEDGEMENTS.md) and [NOTICE](https://github.com/SimpleJev/JevAny/blob/c740121a4532592b09b6272104d5f29a94c232e5/NOTICE) for the full method, model, and asset credits. Other adapters or assets listed there are not necessarily used by this project's mahjong model.

Sources and licenses for this project's Qwen backbone, RiichiEnv, and expert game-record data are documented in [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt), the [dataset documentation](datasets/jev-play-v2/README.md), and the [desktop license](desktop/LICENSE). This project does not include TypeSafe Jev's private weights.
