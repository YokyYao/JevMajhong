# 鸣牌特征Jev Call Reaction v1：继续扩大覆盖，按实际服务范围选权重

**训练完成更新：** 290865完成200步，按事前reaction `transfer_nll`选择第150步；吃碰NLL0.33088799仍差于父模型0.32669174。选中权重新增2400条，整轮实际更新3200条，两者分开计数。290866已通过六条训练/服务概率及六条普通路由核对，并实际进入完整东风对局；尚无Jev Call Reaction v1净得点结论。见[完整曲线、覆盖及训练汇总](../results/2026-11-11-reaction-feature-training/README.md)。

**完整开发更新：** 290866完成768场并审计通过，相对Jev Discard v1平均+339.32点/完整东风，95%整牌山区间[-804.66,1483.31]；强组均值-213.28，放铳率9.46%→11.38%。保留正向信号但不宣称稳定提升，同牌山Jev Call v1父模型对照已实际开始，尚未完成。见[完整开发结果及得点分解](../results/2026-11-11-reaction-feature-development/README.md)。

**父模型对照完成：** Jev Call Reaction v1减Jev Call v1平均-90.63点/完整东风，95%整牌山区间[-695.05,513.80]，未显示续训增益；完整审计通过。条件作业290872按规则跳过独立确认，保留Jev Call v1权重。以下接续登记保留历史记录；新调整见[门清机会课程](closed-opportunity-curriculum-20261111.md)，完整对照见[结果汇总](../results/2026-11-11-reaction-feature-development/README.md)。

## 完整结果与本次决定

Jev Call v1独立确认相对Jev Discard v1为+1045.83点/完整东风，95%整牌山配对区间[321.68,1769.99]。同牌山独立父模型对照也已完成：Jev Call v1减第一阶段为+277.21，区间[-238.43,792.86]，三组对手分别+602.54、+368.16、-139.06。均值为正但增量仍不确定，保留两个阶段，不设固定+250/+500门槛。完整来源与局限见[确认及父模型对照](../results/2026-11-11-feature-continuation-confirmation/README.md)。

继续以Jev Call v1第200步为训练起点。普通弃牌、立直等仍由冻结Jev Discard v1负责，新分支只负责吃/碰/跳过；这不是从旧麻将LoRA初始化，也没有进行Qwen底座的全参数重训。本轮不修改奖励函数，不以增加吃碰次数作为目标。

强组均值下降且放铳率从父模型9.08%升至9.44%，不能宣称防守改善，也不足以断言是哪一类吃碰造成退化。先检验扩大真实专家局面覆盖的效果，同时修正选模范围与实际服务范围不一致的问题；不会依照已看过的确认牌山制作训练标签。

## 数据与真实选模指标

两轮已用6400条请求从原40960条特征TRAIN中按保存的洗牌顺序、实际游标逐轮排除，剩余34560条：吃1810、碰2871、跳过23015、弃牌6792、立直72。仍来自同一1582场TRAIN比赛，不宣称采集了新比赛；也不把初始化前的全部训练历史称为首次见到。

完整validation保留1408条、84场，继续提供整体和各阶段诊断及原有六条训练/服务概率核对。另原样提取其中reaction的1024条、33场，吃74、碰114、跳过836，生成单独的只读验证suite。它是已用验证的子集，**不是独立测试集**；封存test不打开。训练与验证整场交集为0，保留原标签、选项顺序与公开状态特征。

采用现有训练器的`eval_transfer_suite`与真实`early_stop_metric="transfer_nll"`，权重选择和早停记录确实读取这个吃碰子集的NLL。完整validation继续记录自己的development NLL，两者分别存储；不替换整体NLL的含义，不改动已冻结的训练器。每次验证会多执行一次1024条吃碰验证，带来相应耗时。

第0步只核对冻结父权重并记录基线，不作为新候选。更新后必须仍有至少一个吃和碰的精确命中，再按真实reaction NLL选一个候选；不要求先超过父模型NLL才允许一次新得点评估。NLL用于控制候选数量，最终能力仍由完整东风净得点衡量。

## 执行阶段与决策

1. 核对两个完整独立审计、逐场哈希与父权重身份，生成新的不可覆盖数据和运行目录；验证所有冻结来源。加载时逐张量核对父LoRA与动作头，使用新AdamW/OneCycle，而非恢复旧优化器。
2. 新seed、batch1、累积16，LoRA峰值1e-5、头5e-5不变，最多新增200步，即再用3200条请求。每50步验证，patience4；全部34560条必须通过token准入，拒绝0。训练阶段有四小时资源上限，不把分配时长当作必须训练时长。
3. 选择更新后的合格候选后，在10050000–10050031共32个新逻辑牌山、四座、三组固定对手下，与Jev Discard v1各384场。先核对原有六条训练/服务概率、六条普通路由不变，再审计完整计分流水和按整牌山聚合的配对统计。
4. 对Jev Discard v1完整均值为正时，复用候选384场，只新增同牌山Jev Call v1父模型384场，回答本轮续训是否带来增量。普通决策和强组对手都保持同一个Jev Discard v1，不混用不同牌山的均值。
5. 对Jev Discard v1及同牌山父模型的两个完整均值均为正时，才考虑同一候选进入预留10060000–10060127的128牌山独立确认；不因小幅正向或区间跨零自动否决。确认执行前需检查所有身份与完整审计；下述条件接续作业只预排检查和条件执行，不代表已经开始或完成确认。
6. 任一完整均值非正则不盲目延长同一候选，保留已有父权重，依据具体漏吃/误鸣、门清变化及和牌/放铳/流局得点构成调整下一方案。强组负均值和区间都必须公开报告，不把整体正均值解释为所有对手都改善。

阶段成功的证据是新牌山完整对局及其审计，不是单纯NLL降低、一致率升高或鸣牌增多。即使开发为正，也须报告独立确认的不确定性；没有真人胜率结论。

## 调度记录

首次作业290857在第0步验证因专用suite缺少题目`src`元数据而失败，尚未更新权重；旧运行已作废，其依赖290859已取消。普通`eval_data`读取器会补齐这些元数据，而suite读取器只校验清单。修复准备入口，让专用suite先经过同一SDK规范化，保持模型输入、选项和标签不变，并用真实benchmark逐条构造1024条统计行核对。

修复采用独立v2数据/运行目录；训练290865已在ws-l1-007的RTX5000 Ada实际启动，GPU预检其他计算进程为0。开发及同牌山父模型对照290866依赖`afterok:290865`，尚未开始对局。新作业第0步已完整通过，整体NLL0.59829335、reaction NLL0.32669174复现父模型，实际选模字段为`transfer_nll`，34560条TRAIN全部准入、拒绝0，已进入参数更新。

### 自动确认接续登记

在Jev Call Reaction v1开发对局尚未开始、没有读取其中途分数时，预排条件作业290872，依赖`afterok:290866`。依赖覆盖完整768场开发、必要时新增的384场父模型对局及全部审计，而非只依赖训练结束。

复用未改动的[条件检查器](https://github.com/YokyYao/JevMajhong/blob/4bdbd79c39071fb523f79bbe1079303c13770a65/scripts/check_continuation_confirmation.py)、[判定逻辑](https://github.com/YokyYao/JevMajhong/blob/4bdbd79c39071fb523f79bbe1079303c13770a65/continuation_confirmation_gate.py)和[接续脚本](https://github.com/YokyYao/JevMajhong/blob/4bdbd79c39071fb523f79bbe1079303c13770a65/feature_continuation_confirmation.slurm)，其哈希分别为`49b56975238d774b2d512a80b8a07f66f61e91c531291e3a1a3eb602adaafc9e`、`747f7a872f81f55f7659c5539c7ef5e0f3adfd829f47a88d8e374a3145493664`、`918694ccc2b3d564be957f598d38df4ee74f3ce43bd0dfa568043636d93a1baf`。

执行前核对准备来源、候选/父模型/Jev Discard v1权重、源文件、注册牌山、两份完整审计和逐场哈希。对Jev Discard v1和同牌山父模型的两个完整平均差均大于0才进入10060000–10060127独立确认；任一非正正常跳过；缺失或身份变化报错，不当作正向或正常非正结果。既有9项条件检查CPU测试已通过，本次未修改这些源码。区间跨零不自动否决小幅正向，也不据此宣称稳定增益。

当前只是依赖等待中的条件作业；后续独立同牌山父模型对照尚未提交，需依据确认结果安排。

## 文件与已完成检查

配置：[训练JSON](https://github.com/YokyYao/JevMajhong/blob/4bdbd79c39071fb523f79bbe1079303c13770a65/feature-reaction-continuation-20261111-v2.json)、[训练TOML](https://github.com/YokyYao/JevMajhong/blob/4bdbd79c39071fb523f79bbe1079303c13770a65/feature-reaction-continuation-20261111-v2.toml)、[开发及父模型配对](https://github.com/YokyYao/JevMajhong/blob/4bdbd79c39071fb523f79bbe1079303c13770a65/feature-reaction-development-20261111-v2.json)、[预留确认](https://github.com/YokyYao/JevMajhong/blob/4bdbd79c39071fb523f79bbe1079303c13770a65/feature-reaction-confirmation-20261111-v2.json)。

[新准备入口](https://github.com/YokyYao/JevMajhong/blob/4bdbd79c39071fb523f79bbe1079303c13770a65/scripts/prepare_reaction_continuation.py)复用已有完整覆盖排除检查，并在训练提交前冻结reaction suite及两个父模型独立审计来源；准备未完成则明确作废新运行。6项CPU测试覆盖标签和选项保留、训练/封存test拒绝、重复ID、缺少吃碰类别、外来反应动作、错误实际指标/数据路径/截断验证。新增真实SDK读取及benchmark行构造回归测试，覆盖缺失元数据的故障。全部1024条实际记录的benchmark契约检查通过。实际SDK的suite校验、配置解析、整场隔离预检查和冻结来源验证已通过。GPU上的初始化、准入、训练及得点尚待运行，不能由CPU检查代替。

训练继续使用`feature_continuation.slurm`和原训练入口，开发及父模型对照使用`feature_continuation_evaluate.slurm`。只公开代码、配置与汇总；新权重、逐场数据、真人牌谱和准备明细保持私有。截图与Windows安装包测试继续暂缓。
