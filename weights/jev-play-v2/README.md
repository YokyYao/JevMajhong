# Jev Play v2

来源为 jev-unified-attackrisk-v8 最终 2,088 步。同一套 LoRA 与动作头处理所有合法动作，不再按吃碰与出牌切换两套模型。冻结基座 Qwen/Qwen3.5-0.8B 需另外下载；固定 revision 见 manifest 和桌面身份文件。

```bash
python tools/prepare_models.py --profile play-v2
```

公开归档只包含推理参数及身份，移除训练路径、优化器及运行状态；LoRA 参数字节与动作头张量均未改变。manifest.json 提供归档、分片及每个文件的校验值。独立模拟尚未证明优于原版安装包，也未证明真人胜率。

来源身份：`5ebce6af3777854cbec55ba29db2dc90be0f959193114bb3154179c5552370fa`。

发布身份：`a6c7c27205fef0647a9990b3b2dcd42edd643cae5068fbeb270d6de226e7098a`。
