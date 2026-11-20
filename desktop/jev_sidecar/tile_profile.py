"""Explicitly selected YOLO weights and a batched RGB tile-face classifier."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


FACE_WIDTH, FACE_HEIGHT = 40, 56
FACE_CLASSES = tuple(f'{rank}{suit}' for suit in 'mpsz' for rank in range(1, 8 if suit == 'z' else 10))
RIVER_ROTATIONS = {'river0': 0, 'river1': 90, 'river2': 180, 'river3': 270}


def face_region(name, rotation):
    """Use region-specific orientation only for the rotations in the training data."""
    return name if name == 'hand' or RIVER_ROTATIONS.get(name) == rotation else ''


@dataclass(frozen=True)
class TileVisionProfile:
    detector: Path
    detector_sha256: str
    classifier: Path
    classifier_sha256: str
    cutoff: float
    margin: float

    @classmethod
    def read(cls, file):
        path = Path(file).resolve()
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if data['schema'] != 1:
            raise ValueError('不支持的牌面识别模型配置版本')
        cutoff, margin = float(data['cutoff']), float(data['margin'])
        if not (0 < cutoff <= 1 and 0 < margin <= 1):
            raise ValueError('牌面识别模型阈值无效')
        return cls(path.parent / data['detector'], data['detector_sha256'],
                   path.parent / data['classifier'], data['classifier_sha256'], cutoff, margin)


def checked_session(path, expected_sha, options):
    import onnxruntime as ort

    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha.lower():
        raise ValueError(f'牌面识别模型校验失败：{path.name}')
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def face_crops(image, box, region):
    left, top, right, bottom = box
    patch = image.crop(box)
    if min(patch.size) < 3:
        return []
    sideways = (right - left) >= (bottom - top) * 1.055
    if not sideways:
        rotations = (180,) if region in ('river1', 'river3') else (0,)
    elif region.startswith('meld'):
        rotations = (270, 90)
    else:
        rotations = (180, 0, 270, 90)
    return [patch.rotate(angle, expand=True).resize((FACE_WIDTH, FACE_HEIGHT), Image.Resampling.BILINEAR)
            for angle in rotations]


def rotate_boxes(boxes, size, angle):
    """Map rectangles to the coordinate system of a rotated image."""
    width, height = size
    if angle == 90:
        return [(top, width - right, bottom, width - left) for left, top, right, bottom in boxes]
    if angle == 270:
        return [(height - bottom, left, height - top, right) for left, top, right, bottom in boxes]
    if angle == 180:
        return [(width - right, height - bottom, width - left, height - top) for left, top, right, bottom in boxes]
    return boxes


class TileFaceClassifier:
    def __init__(self, profile, options):
        self.session = checked_session(profile.classifier, profile.classifier_sha256, options)
        self.cutoff, self.margin = profile.cutoff, profile.margin

    def classify(self, image, boxes, region, allowed):
        crops = [face_crops(image, box, region) for box in boxes]
        flattened = [patch for variants in crops for patch in variants]
        if not flattened:
            return [None] * len(boxes)
        pixels = np.asarray([np.asarray(patch) for patch in flattened], dtype=np.float32)
        tensor = np.ascontiguousarray(pixels.transpose(0, 3, 1, 2) / 255)
        logits = self.session.run(['scores'], {'tiles': tensor})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores = exp / exp.sum(axis=1, keepdims=True)
        if allowed is not None:
            scores[:, [i for i, tile in enumerate(FACE_CLASSES) if tile not in allowed]] = 0
        result = []
        offset = 0
        for variants in crops:
            end = offset + len(variants)
            label = None
            if end > offset:
                best_scores = scores[offset:end].max(axis=0)
                runner_up, best = np.argsort(best_scores)[-2:]
                if best_scores[best] >= self.cutoff and best_scores[best] - best_scores[runner_up] >= self.margin:
                    label = FACE_CLASSES[best]
            result.append(label)
            offset = end
        return result
