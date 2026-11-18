# Windows 桌面工具

[下载 JevMahjong-Setup-0.4.0-win-x64.exe](https://github.com/YokyYao/JevMajhong/releases/download/desktop-v0.4.0/JevMahjong-Setup-0.4.0-win-x64.exe)，双击安装即可。安装程序自动下载并核对运行组件，无需自行安装 Python、pip 或 CUDA Toolkit。使用 NVIDIA GPU 需要本机显卡驱动。

首次启动会下载固定版本底座。之后在本机识别截图和运行推荐；首次使用需在设置中校准牌桌区域。工具可以读取本场数、立直棒数，并区分普通出牌与鸣牌窗口。

## 推荐模型

新版默认 **Jev Play v2**：同一套 LoRA 与动作头处理普通出牌、吃、碰、暗杠、大明杠和加杠。来源为 v8 最终 2,088 步，导出只删除训练元数据，参数值不变。模型偏好数字不能当作和牌概率。

这是按用户要求发布的正式版本，尚未证明优于原版 Jev Play v1 或真人胜率提升。[发布说明](RELEASE_NOTES.md) · [模型身份](../weights/jev-play-v2/README.md)。安装包已通过 Windows 构建及安装后推理检查；旧 0.3.0 版本使用 Discard v1＋Call v1 组合。

源码包含本场数、立直棒识别，以及暗杠／加杠候选和单一选项的点击规划。多种杠法的二次选择界面仍未支持。实际截图和 NVIDIA GPU 效果仍需本机验证。

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

Windows 构建入口为 `desktop/windows/Build-Exe.ps1`。GitHub Windows 工作流支持手动触发；发布分支更新 release-request.json 时会启动构建。构建会执行冻结 EXE、安装后推理和文件校验，详见[开发说明](../docs/development.md)。

桌面许可见 [LICENSE](LICENSE)，第三方说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
