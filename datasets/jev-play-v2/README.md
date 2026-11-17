# Jev Play v2 训练与验证数据

本目录公开 v8 最终2,088步（发布名称 Jev Play v2）实际使用的输入和监督目标。训练33,408条，验证620条；不是新增样本，也没有用评测比赛回填训练。JSONL压缩分片共约8.6 MB，带逐片、归档与还原文件SHA256。

```bash
python tools/prepare_training_data.py
```

输出 `data/jev-play-v2/train.jsonl` 和 `validation.jsonl`。恢复只使用Python标准库，校验字节、行数和训练／验证来源组隔离，拒绝覆盖现有目录。

| 数据 | 训练 | 验证 |
| --- | ---: | ---: |
| 门清出牌 | 15,608 | 332 |
| 副露后出牌 | 8,448 | 80 |
| 吃碰窗口 | 6,728 | 80 |
| 杠窗口 | 2,624 | 128 |
| 合计 | 33,408 | 620 |

训练包含29,625条专家牌谱样本及3,783条模拟标签样本；验证包含320及300条。专家操作是模仿目标；模拟标签来自共同未来下的合法动作比较，并非专家标准答案。标签以本局和牌／速度及放铳风险为目标，v8放铳系数1.5；具体历史方案和独立模拟结果见下方链接。

训练损失中专家样本与模拟样本的目标质量配比为75%／25%，不是简单按样本数量平均。模拟部分按门清／吃碰／杠／副露后出牌40%／30%／20%／10%分配；精确权重见 `label-source-balance.json` 和 `loss-source-tags.json`。训练一个epoch、有效batch16、共2,088步；验证曲线见 `validation-curve.json`。从v7的LoRA和动作头继续训练，并非从随机参数训练基座。v7父权重未包含在这个发布包内，因此本包不能单独实现完整训练复现。

发布处理只把 `_meta.id` 与 `_meta.group_id` 转为确定性SHA256，并压缩JSON空白；state、questions、候选、硬／软标签和源标签保持不变。分组关系及原split保留，训练／验证来源组交叉为0。manifest同时记录发布文件和原始本地文件校验值。只公开整理后的决策样本，不包含玩家账号、原始牌谱、隐藏牌山、后续模拟事件明细或优化器。

专家样本来源：NikkeTryHard 的 [tenhou-houou-mjai](https://huggingface.co/datasets/nikketryhard/tenhou-houou-mjai)，固定revision `6450bb584cef43ede3b514df48e301a832c6818b`；归属 NikkeTryHard 及对应 Tenhou／雀魂牌谱来源。上游声明 CC BY 4.0，这些派生专家样本保留该许可及署名：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本项目将牌谱转换为公开观察与合法决策候选、添加特征及训练标签；模拟样本由项目仿真流程生成。第三方许可不因仓库代码许可而改变。

与此数据对应的训练实现保存在[研究分支固定提交](https://github.com/YokyYao/JevMajhong/tree/2f0c84a3f9c813990da17ef7dc4ba13f7b68979e)，主分支桌面发布不包含该分支全部训练改动。[训练方案](../../docs/experiments/plans/unified-attackrisk-lora-v8-20261115.md)；[独立确认摘要](../../docs/experiments/results/2026-11-12-play-input-audit/unified-attackrisk-v8-confirmation-summary.json)。独立评测的原始局面与逐场明细未包含在训练数据中。
