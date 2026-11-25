# 当前 3B 的实际数据配比与训练路线

本表来自当前已登记并启动的 3B 训练，统计的是实际 TRAIN／VAL 文件和已经使用的样本权重。它描述训练输入，不代表模型已经掌握某种役种，也不是新的胜率结果。

## 实际权重继承路线

固定 Qwen base → 基础策略 608 步 → 鸣牌 388 步 → 已选牌效与攻守 1800 步 → 当前役种与路线选择 2500 步预算 → 后续终局位次训练。当前 3B 的累计起点是 2796 步。冻结 v8 用作效果对照，不作为这条课程的初始化权重。1800 步权重来自十个历史 checkpoint 的共同开发对局比较；这次选择已消耗相应开发数据，不能视为独立评测。

## 数据与更新

- Base：`Qwen/Qwen3.5-0.8B`，固定 revision `2fc06364715b967f1860aea9cf38778875588b17`。
- TRAIN：91,247 条；VAL：24,516 条；25 个来源。完整保留继承数据和已有划分。
- 两轮训练，共 182,494 次样本暴露、2500 次更新；batch 1，梯度累积 73。
- 每个训练 epoch 使用确定的来源混排顺序，混排块 511 条；不是新生成 182,494 个独立局面。
- 学习率 3e-6，决策头学习率 1.5e-5；继承 adapter／head，使用新 optimizer。
- 当前是带回放的监督微调。已有 MC 软目标继续保留；完整对局的 reward、和牌率、前二率和净得点另行评估，不能由训练 loss 直接推断。

## 损失权重

19 类继承数据合计占 80% 损失权重，六类新增来源合计占 20%。新增来源是国士无双、九莲宝灯、四暗刻、大三元、七对及随机起手对照，各占约 3.3333%。这些名称表示数据的生成目标或来源分类，不是每条样本最终和牌的役种。七对来源的 3.3333% 也不等于所有数据中七对相关局面的总比例。

下表的损失比例区别于样本条数比例。标量样本权重按 `loss_mass × 91247 / source_count` 计算，配合固定来源混排使用；它不是测得的梯度贡献比例。

| 来源 | TRAIN 条数 | VAL 条数 | 损失权重 | 每条样本标量权重 |
|---|---:|---:|---:|---:|
| `curriculum2:advanced:legacy:call` | 3,569 | 961 | 2.5153% | 0.643084 |
| `curriculum2:advanced:legacy:kan` | 1,786 | 568 | 2.5153% | 1.285087 |
| `curriculum2:advanced:legacy:post_meld` | 5,590 | 1,220 | 2.5153% | 0.410584 |
| `curriculum2:advanced:teacher:call` | 13,399 | 3,674 | 2.5153% | 0.171294 |
| `curriculum2:advanced:teacher:kan` | 729 | 214 | 2.5153% | 3.148376 |
| `curriculum2:advanced:teacher:post_meld` | 9,352 | 2,501 | 2.5153% | 0.245420 |
| `curriculum2:basic:legacy:plain` | 8,826 | 1,865 | 9.1091% | 0.941738 |
| `curriculum2:basic:legacy:riichi_no` | 272 | 70 | 1.7167% | 5.759011 |
| `curriculum2:basic:legacy:riichi_yes` | 194 | 37 | 1.7167% | 8.074489 |
| `curriculum2:basic:teacher:plain` | 3,106 | 769 | 9.1091% | 2.676040 |
| `curriculum2:basic:teacher:riichi_no` | 812 | 238 | 3.1882% | 3.582664 |
| `curriculum2:basic:teacher:riichi_yes` | 1,075 | 287 | 3.1882% | 2.706161 |
| `curriculum2:followup:teacher:post_reach_discard` | 312 | 75 | 0.8800% | 2.573633 |
| `curriculum3:mc:call_window` | 1,503 | 477 | 2.0000% | 1.214198 |
| `curriculum3:mc:closed_discard` | 717 | 457 | 2.0000% | 2.545244 |
| `curriculum3:mc:kan_window` | 248 | 150 | 2.0000% | 7.358629 |
| `curriculum3:mc:open_discard` | 254 | 100 | 2.0000% | 7.184803 |
| `curriculum3:teacher:closed_discard_under_riichi` | 6,860 | 1,948 | 12.0000% | 1.596157 |
| `curriculum3:teacher:ordinary_completion` | 28,991 | 8,077 | 16.0000% | 0.503588 |
| `curriculum3B:teacher:chiitoitsu` | 732 | 206 | 3.3333% | 4.155146 |
| `curriculum3B:teacher:chuuren` | 677 | 182 | 3.3333% | 4.492713 |
| `curriculum3B:teacher:daisangen` | 730 | 145 | 3.3333% | 4.166530 |
| `curriculum3B:teacher:kokushi` | 488 | 107 | 3.3333% | 6.232719 |
| `curriculum3B:teacher:suuankou` | 663 | 147 | 3.3333% | 4.587582 |
| `curriculum3B:teacher:unselected_random_initial_hand_control` | 362 | 41 | 3.3333% | 8.402118 |

[下载同一份配比 CSV](curriculum-active3B-data-20261125.csv)。

## 验证与效果曲线

在 0／625／1250／1875／2500 步记录完整验证结果和固定开发对局，保存真实 loss、CE、reward、和牌率、前二率与净得点。和牌率按局统计，前二率按完整东风结算统计；各自分母不同。92 个定向役种验证局面属于开发诊断，不能代替自然完整对局胜率。

最终保留评测使用尚未参与调参的 32 个逻辑种子、两种对手配置和四个座位，每个策略 256 场完整东风，共 512 场；候选权重在结果出现前固定。准备了评测入口不代表这些对局已经执行。

后续终局数据完成汇总和配比准备后，才会启动第四阶段。参见[当前训练方案](../plans/curriculum-early3A-direct3B-20261125.md)和[数据完成依赖方案](../plans/curriculum-placement-data-completion-20261125.md)。
