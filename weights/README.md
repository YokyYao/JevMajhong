# 模型目录

统一使用“职责＋版本”命名。版本表示具体参数集合，组合策略单独命名；不再用训练阶段编号作为当前模型名称。

| 模型 ID | 含义 | 参数与用途 |
| --- | --- | --- |
| `jev-supervised-v1` | 监督预训练基座 | 原第2000步模型，供复现及恢复 |
| `jev-discard-v1` | 基础出牌策略 | 从监督模型及校准配方恢复；负责普通决策和杠 |
| `jev-call-v0` | 初始鸣牌特征模型 | 历史版本，已由 v1 接替 |
| `jev-call-v1` | 吃碰策略 | 独立的 LoRA 与动作头；负责吃／碰／跳过 |
| `jev-play-v1` | 当前推荐组合 | Discard v1＋Call v1，共用相同底座版本，分别加载 |
| `jev-tempo-v1` | 原配比快和实验 | 更新两个动作头；底座与 LoRA 冻结，未部署 |
| `jev-tempo-balanced-v1` | 动作配比修正实验 | 五种操作分层加权的后续候选，未部署 |
| `jev-tempo-rank-v1` | 直接动作排序实验 | 纯快和开发得点下降，未采用 |
| `jev-tempo-safe-v1` | 快和与风险奖励实验 | 独立确认−736.46点／场，区间跨零，未采用 |
| `jev-tempo-safe-data-v1` | 新旧数据续训 | 开发未改善，保留记录 |
| `jev-value-v1` | 本局得点主目标实验 | 独立确认−1115.36点／场，区间跨零，未采用 |
| `jev-value-defense-v1` | 副露后立直风险数据与分层训练 | 新数据采集中，尚无该候选权重 |
| `jev-call-data-v2-s050` 等 | 新比赛数据续训候选 | 后缀为新增更新步数；不是当前发布模型 |

两套已发布推理参数都使用 `Qwen/Qwen3.5-0.8B`，固定版本 `2fc06364715b967f1860aea9cf38778875588b17`。组合会按决策类型调用一个策略，不会为每次推荐都执行两套模型。

## 恢复

```bash
python tools/prepare_models.py
```

恢复到 `models/jev-discard-v1` 和 `models/jev-call-v1`。程序核对归档、文件及动作头张量，拒绝覆盖已有目录。冻结的底座需另行下载，桌面程序会自动准备它。

公开参数位于 [Jev Supervised v1](jev-supervised-v1/README.md)、[Jev Call v1](jev-call-v1/README.md)；Jev Discard v1 使用[恢复配方](jev-discard-v1/recipe.json)。目录迁移和命名调整不改变模型参数。中间第600步快照已移出当前分支，可从历史提交追溯。

## 历史名称对照

| 历史叫法 | 当前名称 |
| --- | --- |
| 旧 B、旧B、calibration-02 | Jev Discard v1 |
| 第二阶段第200步、feature-stage2-step200 | Jev Call v1 |
| 旧B＋第二阶段组合 | Jev Play v1 |
| 新奖励双动作头实验 | Jev Tempo v1 |

实验记录中的已确认模型名称已同步规范，数值与结论保留；流程步骤中的“阶段”仍按当时的实验含义解释。旧名称对应关系用于阅读历史提交。完整机器可读身份见 [catalog.json](catalog.json)。
