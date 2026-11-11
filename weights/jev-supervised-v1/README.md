# Jev Supervised v1

原第2000步监督预训练参数。保留该完整归档供历史复现和恢复 Jev Discard v1；不是当前推荐组合的全部参数。归档摘要与文件清单见 `manifest.json`。

只恢复本模型：

```bash
python tools/restore_weights.py weights/jev-supervised-v1/manifest.json --out models/jev-supervised-v1
```

桌面组合请使用 `python tools/prepare_models.py`。
