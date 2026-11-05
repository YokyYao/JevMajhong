# Windows EXE 0.2.0 发布验证

[下载 Setup.exe](https://github.com/YokyYao/JevMajhong/releases/download/desktop-v0.2.0/JevMahjong-Setup-0.2.0-win-x64.exe)并双击安装，随后从桌面快捷方式启动。无需安装Python、pip或CUDA Toolkit；本机需要兼容的NVIDIA显卡与驱动。首次在界面中校准牌桌和牌面皮肤。

安装程序为2,579,186字节（约2.46MiB），自动从同一Release下载四个独立运行组件ZIP，总计3,085,294,575字节（约3.09GB），并按编译进EXE的SHA-256校验后安装。组件解压约4.82GB，含Python、CUDA版PyTorch、识别模型、中文字体和已确认推理参数；首次启动另行下载固定版本的Qwen底座。安装与首次启动需要联网，建议预留约12GB磁盘空间。用户无需手动下载或解压Runtime ZIP。

原完整便携ZIP为2,969,438,698字节，超过GitHub单文件2GiB限制，因此采用自动下载多个独立组件的安装方式。下载组件失败或校验失败会停止安装。程序安装在当前用户目录并提供卸载入口。

| 检查 | 结果 |
| --- | --- |
| Windows界面与牌理检查 | [通过](https://github.com/YokyYao/JevMajhong/actions/runs/42057992567) |
| 实际冻结EXE：界面、ONNX识别器与CPU模型推理 | 通过；移除Python环境变量及PATH条目 |
| 安装程序：本地HTTP下载相同组件，安装后实际模型推理 | [通过](https://github.com/YokyYao/JevMajhong/actions/runs/42057992621) |
| 公开下载：校验Setup、从真实GitHub地址安装、空HF缓存首次模型下载及推理 | [通过](https://github.com/YokyYao/JevMajhong/actions/runs/42060264389) |
| 四个运行组件的GitHub资产摘要与构建SHA-256 | 全部一致；每个文件低于2GiB |

两次安装后的真实模型测试均报告`frozen_exe/gui/onnx/model=true`，在多个受约束候选中选择`4z`。公开下载测试未安装或调用外部Python，也没有提供预下载的底座路径，测试从空缓存开始。

发布源码提交：`76e06a5fd4f61d13fbd0707a286072da48646d94`。安装程序SHA-256：`2bf6d42293e3bb56109bd916ef219d912b73a63daaf0e0664c63ca1f293fbc16`。其他文件摘要见同Release的`SHA256SUMS.txt`与`runtime-manifest.json`。

推理模型仍为已确认的256维校正版2。发布时从原公开2000步恢复冻结输入，再重建评分头；张量指纹`ed19999b02abb0a72f3aed0d5f8cc42d9ab04b68f555fff1ba2135de5b462683`及16项运行配置完全校验。只发布运行所需参数，没有加入优化器、训练逐场、特征或梯度诊断。

Windows托管测试机没有NVIDIA GPU，以上实际模型检查使用CPU。既有Linux NVIDIA推理已通过，Windows显卡速度、游戏截图与皮肤识别仍需本机实测；此发布验证不代表真人雀魂胜率验证。许可证、上游来源及完整操作见[桌面说明](../../desktop/README.md)。
