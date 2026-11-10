# Mahjong recommender with JevAny

**模型优先：** [公开鸣牌牌效与新增专家数据训练](plans/feature-expert-training-20261110.md)已启动：40960条原生决策、1582场来源比赛，包含新吃2091、碰3385，并保留普通弃牌样本；每50步验证、首阶段最多200步。旧的部分碰牌检查点同时在768场完整东风对照中评估，普通出牌保持冻结旧B。本轮有合格候选后另用全新牌山评估，尚无新净得点提升结论；截图适配和安装包测试继续暂缓。

**攻略驱动的下一阶段：** 已阅读用户提供合集中的得点、立直、副露、防守与东风策略五篇关键正文，将34项公开状态协变量与配对终局得分学习结合；32个新训练牌山的512条单次鸣牌标签已完成，整牌山分组验证未超过零预测基线，已停止本候选推进。目标仍是净得点，不给吃碰固定奖励，尚无新提升结论。见[方案与适用范围](plans/guide-informed-point-policy-20261110.md)。

**2026-11-10 吃碰试验：** 新自然数据与更强教师约束试验已完成768场开发评估并通过独立审计；相对旧模型两组平均净得点为-497.66、-961.72，区间均跨零，没有稳定改善证据，不部署。此前正向开发信号未在独立复验首组保持，已停止剩余确认。后续转向[单次鸣牌的终局得分标签](plans/call-point-counterfactual-foundation.md)，其一牌山GPU联调已通过，尚未正式拟合；[计分核对已完成](reports/2026-11-10-call-scoring-audit/README.md)：48场精确重放、219次和牌支付额全部通过独立库核对，包含11次副露和牌；未发现计分差额。见[新数据方案与完整结果](plans/call-natural-query-20261110.md)。[根因诊断](reports/2026-11-09-call-policy-diagnosis/README.md)确认回放候选曾泄漏答案；不要把 consumed 改成错误的三张来强迫模型鸣牌。

**2026-11-09 桌面源码调试版与服务器交接：** 后台窗口截图、自动推荐/点击、立直两步确认、启动预热和两条独立策略说明已加入当前源码。吃碰偏向跳过的问题尚未修复或训练，复现输入、已确认的接口差异、模型标识及训练验证建议见[服务器交接说明](plans/desktop-call-policy-server-handoff.md)。本次没有新训练权重；主分支推送会触发已有自动打包流程，安装包以工作流实际结果为准，已安装的旧 EXE 不会自动获得源码改动。

[下载 Windows 安装程序（EXE）](https://github.com/YokyYao/JevMajhong/releases/download/desktop-v0.2.1/JevMahjong-Setup-0.2.1-win-x64.exe)：双击后自动下载、校验并安装运行组件，无需自行安装Python或配置环境。首次启动自动下载固定版本底座，之后本地截图与本机NVIDIA GPU推荐；首次需在界面中校准牌桌区域。0.2.1会自动读取本场与立直棒并更新推荐，不增加计数识别错误提示；已通过Windows冻结EXE和安装后计数OCR/模型推理检查，见[本次更新与验证](reports/2026-11-10-desktop-counters/README.md)及[使用说明](desktop/README.md)。

基于 JevAny 的四人立直麻将训练与离线推荐原型。用高段位雀魂牌谱训练合法动作评分器，并在四人模拟环境中检查一位率、平均顺位和末位风险。

**统计更正：此前连续原始种子的半庄会共享移位后的牌山流，不能直接视为独立样本。历史一位率保留为对应牌山上的描述，相关独立置信区间解释撤回。新评估默认采用隔离种子，详见[牌山种子独立性更正](reports/seed-independence-erratum.md)。**

**最新结果（2026-10-30）：256维得分校正在独立大样本仿真中相对原2000步提高96.29点/场。** 全新512个独立牌山、每模型2048场，配对95%区间[13.22,179.36]，通过预设净得点改善判据，独立审计通过。一位率区间仍含0，不能宣称雀魂段位或真人胜率提高。已确认的校正参数随Windows发布版公开，也可显式通过推荐CLI使用；本次对照是原2000步模型，默认规则策略未切换。详见[固定候选大样本复验](reports/2026-10-30-diagonal-fixed-replication/README.md)。

第一轮2000步训练、完整验证和两组历史仿真均已完成。完整验证动作一致率为67.68%；旧种子方法下，纯模型一位率25.00%，牌效防守约束版本31.25%，各128场。这些对局的独立性假设已撤回，不能据其置信区间判定稳定改善。详见[第一轮结果及逐场记录](reports/2026-10-28-training/README.md)。

已发布[最终第 2000 步权重](checkpoints/step-002000/README.md)，与本轮结果报告评估的模型一致；第 600 步快照仍保留。检查点分片不包含原始牌谱、冻结的 Qwen 底座、虚拟环境或用户凭据，也不提供雀魂客户端接入。

下一轮的数据扩充、弃牌采样、训练参数、选模与独立仿真判据见[第二轮训练方案](plans/round-2-training.md)。方案中的指标门槛是预设实验协议，不是已完成结果。

第二轮监督训练已在1200步早停，400、800、1200步检查点的实战对照均已完成，均落后于旧模型，详见[第二轮实战结果](reports/2026-10-29-round2-strength/README.md)。整体动作一致率仅作诊断，不作为实战能力的替代指标。

[评分头强化学习短程试验](plans/head-rl-pilot.md)已在第32次更新按无进展门槛自动停止，保存9个更新进度。修正种子后的同批64场开发对局中，旧模型与更新32的一位率均为28.13%，平均净得点分别为+3387.50和+2776.56；未达到45%及相对增加5个百分点的目标，没有进入独立确认或替换默认策略。完整记录和后续奖励设计建议见[本轮结果报告](reports/2026-10-29-head-rl-pilot/README.md)。开发筛查不足以证明旧模型已很强或到达上限。

[四人东风前二率训练](plans/top2-east-training.md)已按用户要求在第8次参数更新后取消，保留0、4、8次检查点；第8次开发验证未完成，不报告整体提升结论。原定前二奖励及阶段门槛保留在方案中以便复现。

[四人东风净得点奖励试验](plans/score-east-training.md)已完成16次更新并自动停止。每个检查点64场同批开发对局中，旧模型平均净得点+2068.75，更新8和16均为+2084.375，仅增加15.625点/场；未达到目标，没有进入独立确认，默认策略未替换。奖励和5090个决策回报已核对，详见[结果与逐场记录](reports/2026-10-29-score-east/README.md)。

已完成[补充原因分析](reports/2026-10-29-score-diagnosis/README.md)。随后[配对轨迹留一基线试验](plans/score-east-rloo.md)也在16次更新无进展停止：同批旧模型平均净得点-634.375，候选-710.9375，相对下降76.5625点/场，未进入确认。

[冻结旧模型对照与校准学习率试验](plans/score-east-reference.md)已完成独立确认：同批512场中旧模型平均净得点+977.73、候选+1172.85，提高195.12点/场，配对95%区间[-316.19,706.42]，未达到+500且区间下界大于0的目标。开发+601.56未在独立确认中保持，默认策略未更换；详见[完整结果和审计](reports/2026-10-29-score-reference/README.md)。

[扩大牌山覆盖试验](plans/score-east-coverage.md)在16次更新回退后早停，保留开发较好的更新8。随后[单候选独立确认](plans/score-east-coverage-followup.md)双方各512场：原模型平均净得点+691.80，更新8为+943.75，提高251.95点/场，配对95%区间[-213.16,717.07]，仍未达+500及区间下界大于0目标。原始记录核验通过，详见[完整开发、独立确认和训练记录](reports/2026-10-29-score-coverage/README.md)。

[从更新8继续的小步训练](plans/score-east-carry8.md)完成512场新候选对局及256场冻结对照，在本轮第12次开发相对更新8提高468.75点/场后冻结。最终确认双方各512场，相对原2000步模型-53.91点/场，配对95%区间[-444.38,336.57]，未达标，默认策略未替换。详见[开发与最终结果摘要](reports/2026-10-29-score-carry8/README.md)。

[回报基线校准试验](plans/score-east-calibrated-cache.md)已在16次无进展早停：相对原模型的开发净得点变化为-157.81/-381.25点。回报与梯度加权残差均降低，但没有带来开发得分提高；原4次更新可精确重现，进一步梯度加权改善很小或略差，未扩展新的训练分支。详见[完整结果与诊断](reports/2026-10-29-reference-calibration/README.md)。更新12的最终确认也未达标。[单步反事实得分训练](plans/score-east-single-decision.md)已在更新2首次开发检查相对阶段参考+262.89点/场后冻结，配对95%区间[-268.94,794.72]仍含0。512个样本的单步奖励与采样核验通过；最终双方各512场相对原2000步模型-85.55点/场，区间[-472.05,300.96]，未达标。见[阶段结果摘要](reports/2026-10-29-score-single-decision/README.md)，本轮逐场明细与权重留在本地。

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

[得分差加权的参考相对排序试验](plans/score-east-relative-ranking.md)在同一批单步反事实数据上改用成对排序损失，针对多数有利替代动作也被压低的问题；阶段门槛要求超过失败PPO更新2的开发均值250点，最终仍使用全新牌山对照原2000步模型。训练起点第三组补测只作事后诊断，不重用为独立通过证据。 作业254438已完成8次更新，保留开发净得点最好的更新6（相对阶段参考+444.53点/场，区间仍含0），随后补充新牌山续训，见[训练摘要](reports/2026-10-29-score-relative-ranking/README.md)。

后续按[增量续训规则](plans/score-east-incremental-continuation.md)保留正向检查点并补充新牌山，+250/+500仅作期望目标。宣布有效提高仍需原2000步模型对照、全新牌山和配对区间下界大于0；历史结果按原协议保留。

[新牌山增量续训](reports/2026-10-30-score-incremental/README.md)的第一批2048个单步样本已完成并审计通过；更新4/8完整开发相对本轮冻结起点-82.42/-53.91点/场，区间均含0，按两次未创新高停止；保留原排序更新6并进行新的独立确认。保留最好检查点，小幅正向信号不受固定增幅否决。

[保留排序更新6的独立确认方案](plans/ranking6-preserved-confirmation.md)固定一个候选，在全新4202牌山与原2000步模型比较。+500是期望目标，改善证据以配对区间下界大于0为准，历史结果不回写。

推荐入口已对齐当前独立评估的未合并LoRA加载方式；`recommend_state.py --guarded`使用同一原生策略。配置回归测试已通过，独立确认结束后另做真实观测的端到端一致性检查，不以接口检查代替净得点验证。
