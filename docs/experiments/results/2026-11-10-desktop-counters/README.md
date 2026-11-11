# 本场与立直棒自动读取

Windows工具的截图流程现可读取宝牌栏下方的两项计数：左侧立直棒、右侧本场。识别结果自动填入牌桌信息并进入Jev请求，变化时作废旧推荐、重新分析；包括从非零恢复为0。按用户要求，不增加计数识别错误提示，未读到数字的帧保留已有值，不伪造0。

默认适配雀魂左上角面板，也可使用已有框选机制选择“立直棒与本场（整行）”适配不同布局。计数不送入麻将牌面检测器，避免把点棒误当成牌。OCR复用既有本机RapidOCR及已打包模型，不增加在线服务或额外安装步骤。

## 已完成验证

- 现有桌面、图像、计数与归档共40项测试通过；覆盖左右对应、OCR文本框逆序、两位数、归零、短暂空帧、UI更新后旧推荐失效。
- 用户提供的攻略截图按实际计数区域读取，结果为本场0、立直棒0；已有完整公开牌桌截图使用默认区域也得到0/0。两张原图及逐帧内容不纳入本次公开提交。
- 用真实RapidOCR运行生成的数字样例，立直棒/本场分别2/3、12/7、0/0均正确。生成样例不能代替实际游戏识别准确率评测。
- Linux桌面运行自检通过GUI、牌面ONNX和实际计数OCR；本场3与立直棒2已核对进入模型请求。没有把Linux测试冒称Windows EXE验证。

## Windows发布

[0.2.1安装包](https://github.com/YokyYao/JevMajhong/releases/download/desktop-v0.2.1/JevMahjong-Setup-0.2.1-win-x64.exe)已公开发布，来源提交`433a9d6ea8e3115f0871f39fec963bd0fc11c4a4`。安装程序自动获取并校验运行组件，不需要自行配置Python。原0.2.0包不包含本次新增功能。

[Windows界面检查](https://github.com/YokyYao/JevMajhong/actions/runs/42801396189)及[构建、冻结EXE、安装后模型和OCR检查](https://github.com/YokyYao/JevMajhong/actions/runs/42801395792)均通过；后者已确认从PATH移除Python后仍可运行。公开Setup下载与SHA256检查也已通过，[使用公开URL安装及首次启动的复验](https://github.com/YokyYao/JevMajhong/actions/runs/42803872372)仍在运行，尚不将该完整复验记为成功。

Setup SHA256：`dfa98fe1be91d18083ffc26ebf635b5d2e9975080b14df64e053dc9ea6a2f6ef`。

现有推荐模型为已确认校正版2，本次更新是计数读取与请求刷新。
