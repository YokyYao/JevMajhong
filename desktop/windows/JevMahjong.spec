from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parents[1]
datas, binaries, hidden = [], [], []
for package in ['onnxruntime', 'safetensors', 'tokenizers', 'rapidocr_onnxruntime', 'windows_capture']:
    data, binary, modules = collect_all(package)
    datas += data
    binaries += binary
    hidden += modules
for package in ['transformers', 'peft', 'jevany', 'accelerate']:
    datas += collect_data_files(package, include_py_files=True)
for distribution in ['transformers', 'peft', 'accelerate', 'torch', 'jevany', 'huggingface-hub']:
    datas += copy_metadata(distribution, recursive=True)
hidden += collect_submodules('mahjong')
hidden += collect_submodules('jevany.backbones')
hidden += ['jevany.api', 'jevany.model', 'jevany.checkpoint', 'jevany.inference',
           'transformers.models.qwen3_5.modeling_qwen3_5', 'transformers.models.qwen3_5.configuration_qwen3_5',
           'transformers.models.qwen2.tokenization_qwen2', 'transformers.models.qwen2.tokenization_qwen2_fast',
           'transformers.generation', 'peft.tuners.lora', 'torch._C']
datas += [(str(root / 'desktop/jev_sidecar/identity_data.json'), 'desktop/jev_sidecar')]
datas += [(str(root / 'desktop/jev_sidecar/stage2_identity.json'), 'desktop/jev_sidecar')]
a = Analysis([str(root / 'desktop/launcher.py')], pathex=[str(root)],
             binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=['tkinter', 'matplotlib', 'IPython', 'pytest', 'tensorboard', 'wandb',
                       'tensorflow', 'jax', 'datasets', 'triton'])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='JevMahjong', debug=False, strip=False, upx=False, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='JevMahjong')
