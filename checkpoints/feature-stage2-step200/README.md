# 第二阶段第200步：推理权重

本目录提供用户指定公开的第二阶段推理检查点，模型标识`feature-stage2-step200`。它包含Qwen3.5-0.8B对应的rank16 LoRA、256维指针动作头、分词器及运行配置，需搭配固定底座版本`2fc06364715b967f1860aea9cf38778875588b17`使用；不是重新训练得到的完整0.8B底座，也不是只有一个独立头。

| 文件 | SHA-256 |
| --- | --- |
| `adapter_model.safetensors` | `7788cff96a921f1358f0e68fa6ee16f69782174c106e98c6a11008acdff0becb` |
| `head.pt` | `8f3421dca9a700c63d2a7b1694a81c80a0ef9f29c263a13a75fb13e426452a04` |

该阶段也使用普通出牌监督数据。此前确认的效果来自**旧B负责普通/罕见决策、该检查点负责chi/pon/pass**的组合；不是该检查点单独负责全部决策的结果。[独立完整东风模拟确认](../../reports/2026-11-11-feature-continuation-confirmation/README.md)相对旧B平均+1045.83点/场，95%整牌山配对区间[321.68,1769.99]。固定规则对手的结果不能当作截图工具在真人雀魂中的胜率。单模型与组合的[同牌山补充对照](../../plans/stage2-full-vs-hybrid-20261111.md)已完成：单模型减组合−1521.61点/场，95%区间[−3401.35,358.12]，未显示替换收益，仍保留组合。

`manifest.json`列出全部6个推理文件及12个二进制传输分片的大小和哈希。分片总计46,485,419字节，恢复后逐文件校验。仅公开推理文件；没有优化器、训练游标、真人牌谱或模拟逐场记录。

源码开发者从仓库根目录恢复：

```bash
python scripts/restore_checkpoint.py checkpoints/feature-stage2-step200/manifest.json --out models/feature-stage2
```

桌面构建入口会自动恢复本检查点和旧B，安装用户无需执行此命令。[0.3.0 Windows EXE](https://github.com/YokyYao/JevMajhong/releases/download/desktop-v0.3.0/JevMahjong-Setup-0.3.0-win-x64.exe)已发布，实际冻结程序、安装后CPU推理与自动组件下载安装检查均通过。两套策略采用相同底座版本，目前分别加载；Windows NVIDIA显卡与截图实战效果仍需本机测试。旧0.2.1 EXE不会自动读取这个新模型。底座、JevAny及数据来源说明沿用仓库第三方说明。
