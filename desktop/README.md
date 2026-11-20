# Windows 桌面工具

[下载最新版 Windows 安装程序](https://github.com/YokyYao/JevMajhong/releases/latest)，双击安装即可。安装程序自动下载并核对运行组件，无需自行安装 Python、pip 或 CUDA Toolkit。使用 NVIDIA GPU 需要本机显卡驱动。

升级时直接运行新版安装程序并使用原安装目录，**不要先卸载旧版**。安装器会校验并复用已有 Python／PyTorch／CUDA 运行环境；程序与模型分别更新，模型相同时也不重复下载。首次安装、换到空目录或运行环境缺失／损坏时，才下载完整运行组件。截图校准及已下载的 Qwen 基座缓存会保留。

首次启动会下载固定版本底座。之后在本机识别截图和运行推荐；首次使用需在设置中校准牌桌区域。工具可以读取本场数、立直棒数，并区分普通出牌与鸣牌窗口。

要连续打多场，先设置总场数，再勾选「自动执行推荐」和「连打」。可以手动进入第一场，也可以从当前结算页或匹配确认弹窗接续。助手按当前画面识别按钮，已结束的上一场不计入场数，达到设定场数后停在结算页。按 Esc 可以停止自动点击和连打。

## 推荐模型

新版默认 **Jev Play v2**：同一套 LoRA 与动作头处理普通出牌、吃、碰、暗杠、大明杠和加杠。来源为 v8 最终 2,088 步，导出只删除训练元数据，参数值不变。模型偏好数字不能当作和牌概率。

这是按用户要求发布的正式版本，尚未证明优于原版 Jev Play v1 或真人胜率提升。[发布说明](RELEASE_NOTES.md) · [模型身份](../weights/jev-play-v2/README.md)。安装包已通过 Windows 构建及安装后推理检查；旧 0.3.0 版本使用 Discard v1＋Call v1 组合。

源码包含本场数、立直棒识别，以及暗杠／加杠候选和单一选项的点击规划。多种杠法的二次选择界面仍未支持。本机已使用 NVIDIA GPU 和实际牌桌截图完成两场真人对局验收。

## 从源码运行

```bash
python -m pip install -r desktop/requirements-ui.txt
python -m pip install -r desktop/requirements-model.txt
python tools/prepare_models.py
python desktop/fetch_assets.py
python desktop/launcher.py
```

界面及牌理功能可先使用，模型模式需要已恢复的本地参数。推荐结果仍依赖当前截图是否提供完整牌河、副露和场况。

本地训练的牌面识别候选可用 `python -m desktop.jev_sidecar.app --tile-vision-profile <profile.json>` 试用。配置文件指定 YOLO 与牌面分类器的 ONNX 文件、各自的 SHA-256，以及分类置信度和分差阈值。两份权重会在启用时校验。

## 构建

Windows 构建入口为 `desktop/windows/Build-Exe.ps1`。GitHub Windows 工作流支持手动触发；发布分支更新 release-request.json 时会启动构建。构建会执行冻结 EXE、安装后推理和文件校验，详见[开发说明](../docs/development.md)。

桌面许可见 [LICENSE](LICENSE)，第三方说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

维护说明：当前首次安装／修复以 desktop-v0.4.0 的四个 Runtime 附件为基线，请保留这些已发布附件。构建使用锁定运行库版本；新版程序与基线不同的文件进入 Application 更新包，模型单独打包。发布前会实际测试 0.4.0 覆盖升级、全新安装、重复安装及损坏修复，并记录下载请求，禁止完整环境重复下载。更换运行库时须同步维护基线和升级测试。
