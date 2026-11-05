# Desktop component notices

The new desktop component is distributed under AGPL-3.0; see LICENSE. This directory's license does not relicense unrelated training files.

* Tile detector: [ssttkkl/mahjong-utils-app](https://github.com/ssttkkl/mahjong-utils-app), pinned revision `eb360862738a5d4edcaaf5fdb2de566a30a8a78e`, `mahjong-detector/src/desktopMain/resources/best.fp16.onnx`, SHA256 `70caa1c6b6809345695cf3a1794a97772edb055356afccdf8e441f14e025797e`. The repository code has an MIT license; the **model's embedded metadata identifies its weights as AGPL-3.0**, which is retained here. The Python pre/postprocessing is independently implemented for this component.
* JevAny: [SimpleJev/JevAny](https://github.com/SimpleJev/JevAny), pinned revision `c740121a4532592b09b6272104d5f29a94c232e5`, Apache-2.0. Installed from its source archive, including upstream notices.
* Qwen base: `Qwen/Qwen3.5-0.8B`, revision `2fc06364715b967f1860aea9cf38778875588b17`, downloaded by the existing checkpoint loader. Base-model upstream terms remain applicable.
* Noto Sans SC: [Google Fonts](https://github.com/google/fonts/tree/fbe81817505e8aeadc53cc7a38a2b76fb57c333e/ofl/notosanssc), SIL Open Font License; text included in `licenses/OFL-NotoSansSC.txt`. Font SHA256 `a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da`.
* PySide6/Qt: LGPL-3.0/GPL/commercial upstream distribution. Pillow, ONNX Runtime, PyTorch, Transformers, PEFT and the MahjongRepository mahjong library are installed as dependencies with their upstream licenses. The Windows source bundle contains no repackaged dependency wheels.
* Recognition development fixture: publicly available `downloads/quehun_reference.jpg` from [jzjzzzzzzz/quehun](https://github.com/jzjzzzzzzz/quehun) at revision `44daa563d9c5e4fff53b09cdfd3a010063bcfc49` (repository Apache-2.0). It was used locally for functional checks and is **not distributed** in the bundle. User screenshots/templates are not shipped.
