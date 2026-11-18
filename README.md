# Jev Mahjong

**简体中文** | [English](README.en.md)

雀魂四人立直麻将的本地出牌与鸣牌推荐工具，以及配套的模拟训练代码。

[下载 Windows 安装程序](https://github.com/YokyYao/JevMajhong/releases/download/desktop-v0.4.2/JevMahjong-Setup-0.4.2-win-x64.exe) · [桌面使用说明](desktop/README.md) · [模型目录](weights/README.md) · [训练与评测](docs/training.md) · [实验记录](docs/experiments/README.md)


Windows 0.4.2 支持复用已安装的运行环境：覆盖安装时只下载变更的程序和模型，保留截图校准及基座缓存；环境缺失或损坏时自动补齐。[升级说明](desktop/README.md)

## 当前模型：Jev Play v2

Windows **0.4.2** 默认使用 **Jev Play v2**。0.4.0、0.4.1 与 0.4.2 使用相同的模型参数；后两次更新主要改进桌面识别、操作和安装升级流程。

| 项目 | 说明 |
| --- | --- |
| 基座与架构 | Qwen3.5-0.8B，基于开放 JevAny 的候选动作决策接口 |
| 训练参数 | 冻结基座，训练同一套 rank16 LoRA 和动作头 |
| 决策范围 | 普通出牌、副露后出牌、立直、和牌（荣和／自摸）、吃、碰、暗杠、大明杠、加杠及跳过 |
| 发布权重 | 实验版本 v8 的最终 2,088 步，正式名称 **Jev Play v2** |
| 运行方式 | Windows 本机截图识别，将结构化公开场况与候选动作交给模型评分 |
| 公开内容 | [推理权重](weights/jev-play-v2/README.md)、[训练／验证数据](datasets/jev-play-v2/README.md)、[历史训练实现](https://github.com/YokyYao/JevMajhong/tree/2f0c84a3f9c813990da17ef7dc4ba13f7b68979e) |

### 训练方式

Jev Play v2 将专家牌谱模仿与模拟动作比较结合。从 v7 的 LoRA 和动作头继续训练，在本轮 **33,408 条训练位置**上完整训练一轮，有效 batch 为 16，共 **2,088 次更新**；另有 **620 条验证位置**，训练与验证按来源组隔离。2,088 是本轮步数，不是模型从基座开始的总训练步数。

| 决策类型 | 训练位置 | 验证位置 |
| --- | ---: | ---: |
| 门清出牌 | 15,608 | 332 |
| 副露后出牌 | 8,448 | 80 |
| 吃碰窗口 | 6,728 | 80 |
| 杠窗口 | 2,624 | 128 |
| **合计** | **33,408** | **620** |

训练集包含 29,625 条专家样本和 3,783 条模拟标签样本。训练损失按专家／模拟 **75%／25%** 分配；模拟部分再按门清／吃碰／杠／副露后出牌 **40%／30%／20%／10%** 分配，这些是损失权重，不是样本数量占比。

模拟标签比较同一局面、共同未来下的候选动作，奖励为 **自己和牌 + 0.25 × 折扣和牌收益 − 1.5 × 是否放铳**，速度折扣为 0.98，净得点项权重为 0；吃、碰、杠本身没有固定加分。模拟标签存在采样与续打策略偏差，不是专家标准答案。[数据与配比](datasets/jev-play-v2/README.md) · [本轮训练方案](docs/experiments/plans/unified-attackrisk-lora-v8-20261115.md)

### 已有模拟结果

四人东风确认评测使用 **32 组新牌山、3 类对手组合、4 个座位**，两种策略共 768 场，各 384 场。对手包含规则策略与历史模型；这些结果只描述该模拟协议。

| 指标 | 实验参照组合 | Jev Play v2 |
| --- | ---: | ---: |
| 一位率（整场第一名） | 28.65% | **33.59%** |
| 前二率 | 56.77% | **64.32%** |
| 四位率 | 22.66% | **19.01%** |
| 平均名次 | 2.3724 | **2.2109** |
| 平均净得点／场 | +1,911.72 | **+3,200.26** |
| 每场等权和牌率 | 25.26% | **27.75%** |

参照为 Discard v1 + Call v1 + Call Tile Win v1 的**实验组合**，与原版 Jev Play v1 安装包不同。净得点差为 **+1,288.54 点／场**，95% 配对区间 **[−298.74，+2,875.83]**；和牌率差为 **+2.50 个百分点**，区间 **[−0.56，+5.55]**。两项区间均跨零，尚未证明稳定提升，也没有证明优于原版安装包或真人胜率提高。768 场来自 32 个独立牌山组，不能当作 768 个独立样本。[完整确认摘要](docs/experiments/results/2026-11-12-play-input-audit/unified-attackrisk-v8-confirmation-summary.json)

### 新增训练要求：末盘追分与守位（待实现）

每局结算后记录四家分数与顺位；末盘落后时根据真实结算、目标分差及连庄／延长条件选择追分路线，领先时兼顾守位。训练目标需覆盖整场终局位次，并同步处理工具的和牌／跳过逻辑。[需求与训练方案](docs/experiments/plans/endgame-placement-20261119.md)

## 目录

| 目录 | 内容 |
| --- | --- |
| `desktop/` | Windows 界面、截图识别与打包配置 |
| `jevmahjong/` | 可复用的牌局状态、策略、数据、训练和评测 |
| `datasets/` | 已公开模型的整理训练／验证数据及校验清单 |
| `weights/` | 模型清单、公开权重和恢复配方 |
| `configs/` | 当前训练／评测模板与环境锁定记录 |
| `tools/`、`cluster/` | 权重恢复、资源准备、打包和通用作业入口 |
| `docs/experiments/` | 历史方案与结果；一次性程序从当前分支移除 |
| `tests/` | 保留的核心与桌面回归检查 |

## 开发

Python 3.12；服务器先安装适合其驱动的 PyTorch，再准备项目环境：

```bash
python -m pip install -e '.[train]'
python tools/fetch_vendor.py
python -m pip install -e 'vendor/JevAny[train,fast]'
python tools/prepare_models.py
python -m jevmahjong --help
```

Jev Play v2 的[训练／验证数据](datasets/jev-play-v2/README.md)和推理权重已整理公开；原始逐场日志、运行目录及其他实验权重仍在本地。快速回归检查与完整对局评测分开，见[开发说明](docs/development.md)。

## 引用与致谢

本项目的决策接口、训练和推理集成基于 **[JevAny](https://github.com/SimpleJev/JevAny)**，作者 **Tianxin Wei**，采用 Apache-2.0 许可。本项目固定使用提交 [c740121](https://github.com/SimpleJev/JevAny/tree/c740121a4532592b09b6272104d5f29a94c232e5)。感谢 JevAny 提供开放的决策模型基础设施。

依据该版本的 [CITATION.cff](https://github.com/SimpleJev/JevAny/blob/c740121a4532592b09b6272104d5f29a94c232e5/CITATION.cff)，使用本项目中基于 JevAny 的工作时，请引用 JevAny，并保留其上游来源署名：

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

JevAny 的基础设施包含由 **Jared Palmer** 创建的 [Kev](https://github.com/jaredpalmer/kev) 的修改代码（Apache-2.0）；其相关组件还署名 [Hugging Face Transformers](https://github.com/huggingface/transformers)、[Cygnet](https://github.com/blockbrain-ai/cygnet-recipe) 与 [NInfer](https://github.com/igorls/ninfer)。完整的方法、模型及素材来源见上游 [ACKNOWLEDGEMENTS.md](https://github.com/SimpleJev/JevAny/blob/c740121a4532592b09b6272104d5f29a94c232e5/ACKNOWLEDGEMENTS.md) 与 [NOTICE](https://github.com/SimpleJev/JevAny/blob/c740121a4532592b09b6272104d5f29a94c232e5/NOTICE)；其中的其他适配器或素材不表示已用于本项目的麻将模型。

本项目使用的 Qwen 基座、RiichiEnv 和专家牌谱数据的来源与许可，见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt)、[数据说明](datasets/jev-play-v2/README.md)及[桌面许可](desktop/LICENSE)。本项目不包含 TypeSafe Jev 的私有权重。
