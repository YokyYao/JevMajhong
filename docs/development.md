# 开发与验证

服务器可复用代码位于 `jevmahjong/`，桌面程序位于 `desktop/`。桌面保留独立的轻量牌面和特征实现，构建 Windows 时无需安装原生 RiichiEnv。

| 模块 | 职责 |
| --- | --- |
| `core/` | 玩家可见状态、隔离牌山种子、计分账本、文件摘要 |
| `policy/` | 模型加载、组合路由、鸣牌特征、动作头替换 |
| `data/` | 牌谱重放修正与整场分组 |
| `training/` | 原生反事实采样、奖励、训练与配比 |
| `evaluation/` | 固定对手、配对完整对局、动作专项验证 |

## 快速检查

```bash
python -m unittest discover -s tests/core -v
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests/desktop -v
```

核心检查无需 GPU 或下载底座。保留信息隔离、种子独立性、计分、奖励方向、完整网格和桌面关键流程；一次性实验的提交器、观察器、校正器测试已移除。这些秒级检查与几十分钟的模拟评测是不同任务。

## 桌面构建

Windows 上安装桌面依赖、PyInstaller、匹配驱动的 PyTorch，恢复权重和识别资源，再运行：

```powershell
./desktop/windows/Build-Exe.ps1 -PythonPath python
```

构建脚本包含推理参数和识别资源，并排除训练数据。手动工作流保留真实冻结 EXE 和安装后检查；默认只上传构建产物，不覆盖已发布版本。

## 实验与发布

通过 `tools/run_experiment.py` 启动时，会把当前可复用代码冻结到忽略的 `runs/code/` 目录，再从该快照运行；后续整理主分支不会改变运行中的代码。每次运行使用新输出目录，记录源码、参数、输入分片摘要。采样器支持 `--resume`，只复用校验通过的完整分片；输出有缺失或损坏时先检查，不静默重写。GPU 作业通过调度器提交，模板见 `cluster/`。

整理前的代码快照固定在 [4bdbd79](https://github.com/YokyYao/JevMajhong/tree/4bdbd79c39071fb523f79bbe1079303c13770a65)。历史报告引用被删除的一次性程序时，链接指向该快照。正在运行的实验继续使用它们注册的原工作区，完成后再迁移后续任务。
