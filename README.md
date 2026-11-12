# Jev Mahjong

雀魂四人立直麻将的本地出牌与鸣牌推荐工具，以及配套的模拟训练代码。

[下载 Windows 安装程序](https://github.com/YokyYao/JevMajhong/releases/download/desktop-v0.3.0/JevMahjong-Setup-0.3.0-win-x64.exe) · [桌面使用说明](desktop/README.md) · [模型目录](weights/README.md) · [训练与评测](docs/training.md) · [实验记录](docs/experiments/README.md)

## 模型

| 名称 | 用途 | 状态 |
| --- | --- | --- |
| **Jev Discard v1** | 普通出牌及杠等决策 | 当前基础策略 |
| **Jev Call v1** | 吃、碰、跳过 | 当前鸣牌策略 |
| **Jev Play v1** | 上面两套策略的组合 | Windows 0.3.0 使用的组合 |
| **Jev Call Uniform v1** | 局面等权的普通吃碰残差组件 | 独立确认−377.34点／场、区间跨零，未采用 |

Jev Play v1 在固定东风模拟中，相对仅使用 Jev Discard v1 提高 **1045.83 点／场**，95%配对区间为 **[321.68，1769.99]**。这项结果属于组合策略；真人对局和截图使用效果仍需验证。[完整记录](docs/experiments/results/2026-11-11-feature-continuation-confirmation/README.md)

实验方案与停止规则见[副露防守训练](docs/experiments/plans/discard-defense-20261112.md)。Call Coverage v1 未显示内部收益改善，已早停；同数据的 Call Uniform v1 只改为局面等权，开发+1331.25点／场但区间跨零，32新牌山768场独立确认为−377.34点／场，区间[−1229.63，474.94]，未采用。当前优先[修复 Play 实战输入与鸣牌策略](docs/experiments/results/2026-11-12-play-input-audit/README.md)。

最新[Windows真人实测与截图证据](docs/experiments/results/2026-11-12-windows-live-test/README.md)：21300分第三名，31次吃碰推荐均跳过；优先复查鸣牌输入与判断，并新增“对手立直后重新比较有役吃碰加速、门清和防守路线”的修复／训练要求，尚未训练部署。

完整实验版本及未采用原因见[模型目录](weights/README.md)；后续候选尚未确认优于Play v1。

权重参数保持原样，模型目录、加载路径和源码界面已使用上述名称。历史报告中的编号对应关系见[模型索引](weights/README.md)。已发布的安装包保持原版本。

## 目录

| 目录 | 内容 |
| --- | --- |
| `desktop/` | Windows 界面、截图识别与打包配置 |
| `jevmahjong/` | 可复用的牌局状态、策略、数据、训练和评测 |
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

训练数据、逐场日志、运行目录和新实验权重保存在本地。快速回归检查与完整对局评测分开，见[开发说明](docs/development.md)。

第三方来源和许可见 [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt) 与 [桌面许可](desktop/LICENSE)。本项目基于开放的 JevAny，不包含 TypeSafe Jev 的私有权重。
