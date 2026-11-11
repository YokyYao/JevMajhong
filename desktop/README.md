# Windows 桌面工具

[下载 JevMahjong-Setup-0.3.0-win-x64.exe](https://github.com/YokyYao/JevMajhong/releases/download/desktop-v0.3.0/JevMahjong-Setup-0.3.0-win-x64.exe)，双击安装即可。安装程序自动下载并核对运行组件，无需自行安装 Python、pip 或 CUDA Toolkit。使用 NVIDIA GPU 需要本机显卡驱动。

首次启动会下载固定版本底座。之后在本机识别截图和运行推荐；首次使用需在设置中校准牌桌区域。工具可以读取本场数、立直棒数，并区分普通出牌与鸣牌窗口。

## 推荐模型

当前发布包对应 **Jev Play v1**：普通决策使用 **Jev Discard v1**，纯吃／碰／跳过使用 **Jev Call v1**。两套参数分别加载，每次决策选择其中一套执行。模型偏好数字不能当作和牌概率。

源码统一使用新名称和目录；0.3.0 安装包保留发布时的界面名称，行为和参数对应上述模型。[身份及历史名称对照](../weights/README.md)

当前截图端能生成大明杠候选，暗杠和加杠候选仍待补齐；有杠候选的决策使用 Discard v1。[操作覆盖检查](../docs/experiments/results/2026-11-11-kan-policy-audit/README.md)

## 从源码运行

```bash
python -m pip install -r desktop/requirements-ui.txt
python -m pip install -r desktop/requirements-model.txt
python tools/prepare_models.py
python desktop/fetch_assets.py
python desktop/launcher.py
```

界面及牌理功能可先使用，模型模式需要已恢复的本地参数。推荐结果仍依赖当前截图是否提供完整牌河、副露和场况。

## 构建

Windows 构建入口为 `desktop/windows/Build-Exe.ps1`。GitHub 的 Windows 构建工作流改为手动触发；仓库整理不会自动重发安装包。构建会执行冻结 EXE、安装后推理和文件校验，详见[开发说明](../docs/development.md)。

桌面许可见 [LICENSE](LICENSE)，第三方说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
