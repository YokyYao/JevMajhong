# Mahjong recommender with JevAny

基于 JevAny 的四人立直麻将训练与离线推荐原型。用高段位雀魂牌谱训练合法动作评分器，并在四人模拟环境中检查一位率、平均顺位和末位风险。

**当前状态：本轮 2000 步训练、完整验证和两组仿真均已完成。** 完整验证动作一致率为 67.68%；纯模型一位率为 25.00%，牌效防守约束版本为 31.25%，各评估 128 场。约束版本有改善信号，但一位率提升未通过预设的置信区间门槛，默认策略仍为规则基线。详见[本轮结果总结及逐场证据](reports/2026-10-28-training/README.md)。

已发布[最终第 2000 步权重](checkpoints/step-002000/README.md)，与本轮结果报告评估的模型一致；第 600 步快照仍保留。检查点分片不包含原始牌谱、冻结的 Qwen 底座、虚拟环境或用户凭据，也不提供雀魂客户端接入。

下一轮的数据扩充、弃牌采样、训练参数、选模与独立仿真判据见[第二轮训练方案](plans/round-2-training.md)。方案中的指标门槛是预设实验协议，不是已完成结果。

第二轮监督训练已在1200步早停，400、800、1200步检查点的实战对照均已完成，均落后于旧模型，详见[第二轮实战结果](reports/2026-10-29-round2-strength/README.md)。整体动作一致率仅作诊断，不作为实战能力的替代指标。

下一阶段采用[评分头强化学习短程试验](plans/head-rl-pilot.md)：冻结旧模型底座与LoRA，以半庄最终第一为奖励，最多64次小批更新，每4次保存、每16次做实战验证。开发目标为一位率至少45%且比同批旧模型高至少5个百分点；未达到阶段进展或出现明显退化就停止并分析。配置和代码已提供，这些目标不是已取得的训练结果。

## 检查点恢复

`checkpoints/step-002000` 包含最终第2000步的 LoRA、动作评分头、tokenizer，以及优化器、调度器和随机状态。为适应上传接口，归档拆为32片，无需Git LFS。每片、整个归档和还原文件均有SHA-256校验。

```bash
python3 scripts/restore_checkpoint.py checkpoints/step-002000/manifest.json \
  --out runs/restored-step-002000
```

安装下述环境后即可加载还原目录。冻结底座从固定版本另行下载。第2000步已经完成原定训练计划；新增训练应创建新配置并用 `init_from` 初始化。仍可用第600步快照在相同单卡 world size、设备类型、数据及兼容环境下恢复其剩余训练。检查点详情和已完成的验证见各自目录下的 README。

## 结构

- `semantic.py`：训练与对局共用的玩家可见状态、合法动作、向听/进张/现物特征。
- `prepare_raw.py`：按雀魂规则重放原始 MJAI 牌谱，生成 JevAny JSONL，按整场比赛隔离数据。
- `semantic-*.toml`：语义预热、完整单卡训练、双卡 DDP 配置。
- `arena.py`：固定初始牌山、换座、与牌效防守基线配对比较，支持评测接续。
- `evaluate.py`：同一上下文限制下的牌谱动作一致率和 NLL。
- `promote.py`：根据对局结果决定候选是否满足当前提升判据。
- `recommend.py`：对一个 JevAny 请求输出 Top-3 动作。
- `recommend_state.py`：对导出的 RiichiEnv 玩家观测推荐合法 MJAI 动作。
- `prepare_jevany.py`、`audit_shard.py`、`mahjong-sft.toml`：早期 NPZ 数据实验，使用不同输入格式，不可与语义模型混用。

## 环境

Python 3.12。先创建独立环境，安装适合目标计算节点驱动的 CUDA PyTorch。旧集群测试版本为 `torch==2.11.0+cu128`，完整依赖记录见 `requirements.lock.txt`；迁移时先核对目标驱动，不要直接复制旧 `.venv`。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/fetch_vendor.py
.venv/bin/python -m pip install -e 'vendor/JevAny[train,fast]'
```

`fetch_vendor.py` 拉取并锁定 JevAny，不复制上游代码进本仓库。配置锁定 `Qwen/Qwen3.5-0.8B` 底座版本；首次训练会下载底座。

CPU 环境即可检查玩家信息隔离和数据分组：

```bash
.venv/bin/python -m unittest test_semantic.py
.venv/bin/python simulate.py --games 2
```

尚未准备真实数据时，数据分组测试会明确跳过。模拟通过不等于模型有牌力。

## 数据准备

原始数据：[`nikketryhard/tenhou-houou-mjai`](https://huggingface.co/datasets/nikketryhard/tenhou-houou-mjai)，固定版本 `6450bb584cef43ede3b514df48e301a832c6818b`。使用其中雀魂王座牌谱，下载前检查上游数据许可。

```bash
.venv/bin/python scripts/download_data.py --year 2019
.venv/bin/python prepare_raw.py data/raw/majsoul-throne-2019.tar \
  --out data/semantic-v1 --train 16000 --validation 512 --test 512 --per-match 64

.venv/bin/python scripts/download_data.py --year 2026
.venv/bin/python prepare_raw.py data/raw/majsoul-throne-2026.tar \
  --out data/semantic-2026-v1 --train 32000 --validation 1024 --test 1024 --per-match 32
```

输出目录须不存在。`provenance.json` 保存来源、原始归档 SHA-256、比赛分组、动作分布与排除原因。目标数据分别来自 250/8/8 场和 1000/32/32 场训练/验证/测试比赛。

所用 RiichiEnv 版本中，回放迭代器的部分初始化事件与真实局面不一致，因此输入只读取观测字段，不读取这些事件。违反可见牌数量守恒的样本会导致整场比赛排除，而不会静默修补。2026 年数据准备排除了 126 场；这可能降低含杠牌谱覆盖，需要继续检查上游重放行为。

## 训练

先运行预热，再运行完整训练。完整模型从预热权重初始化。

```bash
.venv/bin/jevany train --config semantic-warmup.toml --dry-run
.venv/bin/jevany train --config semantic-warmup.toml
.venv/bin/jevany train --config semantic-full.toml
```

单卡有效 batch 为 16。双卡每卡 batch 为 8，总 batch 仍为 16：

```bash
.venv/bin/torchrun --standalone --nproc_per_node=2 \
  --module jevany.train --config semantic-ddp.toml
```

改变 GPU 数量时，不能原样恢复上游保存的优化器进度；使用 `init_from` 创建新的实验，或以相同 world size 使用上游 `--resume`。新实验修改 `out`，避免覆盖已有结果。

Slurm 脚本保留资源规模示例，但不写死旧集群的账号、分区或项目路径。请在目标集群核实配额，并从仓库根目录提交：

```bash
mkdir -p logs
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION warmup.slurm
# 预热完成后，选择单卡或双卡实验之一：
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION full.slurm
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION ddp.slurm
```

上述命令会实际申请 GPU；仅克隆仓库或还原检查点不会启动训练。

## 评测与推荐

在已分配的 GPU 上：

```bash
.venv/bin/python evaluate.py runs/semantic-warmup data/semantic-v1/validation.jsonl \
  --out runs/semantic-warmup/new-validation.json
.venv/bin/python arena.py --checkpoint runs/semantic-warmup \
  --seeds 32 --seed-start 610000 --out runs/new-arena
.venv/bin/python recommend.py runs/semantic-warmup data/semantic-v1/validation.jsonl
```

`arena.py --resume` 接续同配置下已经完成的整场对局。固定种子只固定初始牌山，策略改变后，摸打过程会分叉。换座的四场比赛属于同一个种子分组；顺位差区间按种子聚类计算。

当前基线是保守的牌效防守策略。`--guarded` 版本限制模型的弃牌向听与现物风险，须单独评测，不能假设它一定更强。`promote.py` 的自动选择只支持相对这一基线的判断，不代表雀魂段位。默认推荐入口在没有通过评测的候选时使用基线。

动作概率不是和牌率。语义预热在 512 条验证决策上的一致率为 58.98%、NLL 为 1.190；仅两个独立牌山种子的对局检查不能支持胜率结论。正式评测需冻结模型、指标和对手，使用未参与调参的新牌山，并扩大独立种子数量。旧的预热与完整训练脚本存在评测种子重合，不能把其结果直接作为最终测试证据。

本轮训练从第600步起，每200步对保存的检查点做一次固定验证子集推理：32场验证牌谱各4个决策，共128条。结果输出在对应运行目录的 `periodic-validation/curve.json` 和 `best-validation.json`。此监测不更新参数、不读取测试集，也不会自行停止训练。本轮评估使用最终第2000步模型；验证曲线与完整验证结果见[结果总结](reports/2026-10-28-training/README.md)。

## 来源

第三方项目和数据署名见 `THIRD_PARTY_NOTICES.txt`。本项目采用开放的 JevAny 训练框架，不包含 TypeSafe Jev 的权重或私有实现。
