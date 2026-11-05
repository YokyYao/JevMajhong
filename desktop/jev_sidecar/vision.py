"""Local ONNX Mahjong Soul tile detection, with explicitly calibrated regions."""
import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
import uuid
from threading import RLock

import numpy as np
from PIL import Image, ImageOps

DETECTOR_REVISION = 'eb360862738a5d4edcaaf5fdb2de566a30a8a78e'
DETECTOR_URL = f'https://raw.githubusercontent.com/ssttkkl/mahjong-utils-app/{DETECTOR_REVISION}/mahjong-detector/src/desktopMain/resources/best.fp16.onnx'
DETECTOR_SHA = '70caa1c6b6809345695cf3a1794a97772edb055356afccdf8e441f14e025797e'
HONORS = {'tou': '1z', 'nan': '2z', 'sha': '3z', 'pe': '4z', 'haku': '5z', 'hatsu': '6z', 'chun': '7z'}


@dataclass
class Detection:
    tile: str
    confidence: float
    box: tuple
    source: str = 'ONNX'


def hand_boxes(image):
    """Find separate light tile faces inside a tightly calibrated hand region."""
    array = np.asarray(image.convert('RGB')).astype(float)
    mask = (array.min(2) > 165) & (array.max(2) - array.min(2) < 65)
    rows = np.flatnonzero(mask.mean(1) > .40)
    if not len(rows):
        return []
    top = int(rows[0])
    # Use the blank paper band above the glyph. Full-height projections split
    # dense eight-pin faces into two fragments.
    active = mask[top + 2:min(top + 8, image.height)].mean(0) > .5
    boundaries = np.diff(np.r_[False, active, False].astype(int))
    starts, ends = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
    spans = [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= max(8, image.height * .22)]
    if len(spans) not in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14):
        return []
    result = []
    for a, b in spans:
        ys = np.flatnonzero(mask[:, a:b].mean(1) > .15)
        if len(ys):
            result.append((max(0, a - 2), max(0, int(ys[0]) - 2), min(image.width, b + 2), min(image.height, int(ys[-1]) + 3)))
    return result


class Templates:
    def __init__(self, directory):
        self.directory = Path(directory)
        self._vectors = None
        self._lock = RLock()

    @staticmethod
    def vector(image):
        image = ImageOps.grayscale(image)
        w, h = image.size
        image = image.crop((w * .05, h * .04, w * .95, h * .93)).resize((32, 48))
        vector = np.asarray(image, dtype=float).reshape(-1)
        vector -= vector.mean()
        return vector / max(float(np.linalg.norm(vector)), 1e-8)

    def teach(self, image, boxes, tiles):
        from .tiles import parse
        parse(' '.join(tiles))
        if len(boxes) != len(tiles):
            raise ValueError(f'分割到{len(boxes)}个牌槽，校正手牌有{len(tiles)}张；请收紧手牌框选范围')
        with self._lock:
            for box, tile in zip(boxes, tiles):
                directory = self.directory / tile
                directory.mkdir(parents=True, exist_ok=True)
                image.crop(box).save(directory / (uuid.uuid4().hex + '.png'))
                for old in sorted(directory.glob('*.png'), key=lambda p: p.stat().st_mtime)[:-5]:
                    old.unlink()
            self._vectors = None

    def vectors(self):
        with self._lock:
            if self._vectors is None:
                values = []
                for path in self.directory.glob('*/*.png'):
                    with Image.open(path) as template:
                        values.append((path.parent.name, self.vector(template)))
                self._vectors = values
            return list(self._vectors)

    def classify(self, image):
        vector = self.vector(image)
        scores = {}
        for tile, template in self.vectors():
            similarity = float(vector @ template)
            scores[tile] = max(scores.get(tile, -1), similarity)
        ranked = sorted(scores.items(), key=lambda item: -item[1])
        if not ranked:
            return None
        best = ranked[0]
        margin = best[1] - ranked[1][1] if len(ranked) > 1 else 1
        return best if best[1] >= .97 and margin >= .025 else None


def nms(detections, threshold=.3):
    result = []
    for candidate in sorted(detections, key=lambda d: -d.confidence):
        def overlap(other):
            a, b = candidate.box, other.box
            area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            total = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area
            return area / max(total, 1e-8)
        if all(overlap(other) <= threshold for other in result):
            result.append(candidate)
    return result


def red_five(image, detection):
    tile = detection.tile
    if tile not in ('5m', '5p', '5s'):
        return tile
    a = np.asarray(image.crop(detection.box).convert('RGB'))
    if a.shape[0] < 6 or a.shape[1] < 4:
        return tile
    # Ordinary manzu have a red 萬 below the numeral: inspect only the upper half.
    if tile == '5m':
        a = a[:int(a.shape[0] * .5)]
    r, g, b = [a[:, :, i].astype(float) for i in range(3)]
    red = (r > 90) & (r > g * 1.4) & (r > b * 1.4) & (r - g > 45)
    return '0' + tile[1] if float(red.mean()) > .045 else tile


class TileDetector:
    def __init__(self, path, templates=None):
        self.path, self.session = Path(path), None
        self.loaded_path = None
        self.templates = templates

    def load(self):
        import onnxruntime as ort
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != DETECTOR_SHA:
            raise ValueError('识别模型缺失或校验失败，请运行安装脚本')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(self.path), sess_options=options, providers=['CPUExecutionProvider'])
        self.names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map['names'])
        self.loaded_path = self.path.resolve()

    def detect(self, image, confidence=.50):
        if self.session is None or self.loaded_path != self.path.resolve():
            self.load()
        image = image.convert('RGB')
        width, height = image.size
        gray = ImageOps.autocontrast(ImageOps.grayscale(image)).convert('RGB')
        scale = 640 / max(width, height)
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        dx, dy = (640 - size[0]) // 2, (640 - size[1]) // 2
        padded = Image.new('RGB', (640, 640))
        padded.paste(gray.resize(size, Image.Resampling.BILINEAR), (dx, dy))
        tensor = np.asarray(padded).transpose(2, 0, 1)[None].astype(np.float16) / np.float16(255)
        prediction = self.session.run(None, {self.session.get_inputs()[0].name: tensor})[0][0].T
        scores = prediction[:, 4:]
        classes = scores.argmax(1)
        best = scores[np.arange(len(scores)), classes]
        detections = []
        for row, label, score in zip(prediction[best >= confidence], classes[best >= confidence], best[best >= confidence]):
            xc, yc, w, h = [float(x) for x in row[:4]]
            box = (max(0, (xc - w / 2 - dx) / scale), max(0, (yc - h / 2 - dy) / scale),
                   min(width, (xc + w / 2 - dx) / scale), min(height, (yc + h / 2 - dy) / scale))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            tile = self.names[int(label)]
            tile = HONORS.get(tile, tile)
            detected = Detection(tile, float(score), box)
            detected.tile = red_five(image, detected)
            detections.append(detected)
        return sorted(nms(detections), key=lambda d: (round(d.box[1] / max(8, (d.box[3] - d.box[1]) * .65)), d.box[0]))

    def regions(self, image, regions):
        found = {}
        for name, region in regions.items():
            x, y, w, h = region['rect']
            box = (round(x * image.width), round(y * image.height), round((x + w) * image.width), round((y + h) * image.height))
            crop = image.crop(box)
            rotation = int(region.get('rotation', 0))
            if rotation:
                crop = crop.rotate(rotation, expand=True)
            values = self.detect(crop)
            if name == 'hand':
                boxes = hand_boxes(crop)
                if boxes:
                    segmented = []
                    for box in boxes:
                        match = self.templates.classify(crop.crop(box)) if self.templates else None
                        if match:
                            segmented.append(Detection(match[0], match[1], box, '本地校正模板'))
                            continue
                        possible = [d for d in values if box[0] <= (d.box[0] + d.box[2]) / 2 <= box[2]]
                        if possible:
                            best = max(possible, key=lambda d: d.confidence)
                            confidence = min(best.confidence, .69) if self.templates and self.templates.vectors() else best.confidence
                            segmented.append(Detection(best.tile, confidence, box))
                        else:
                            single = self.detect(crop.crop(box))
                            if single:
                                best = max(single, key=lambda d: d.confidence)
                                confidence = min(best.confidence * .9, .69) if self.templates and self.templates.vectors() else best.confidence * .9
                                segmented.append(Detection(best.tile, confidence, box))
                    if len(segmented) == len(boxes):
                        values = segmented
                values.sort(key=lambda d: d.box[0])
            found[name] = values
        return found
