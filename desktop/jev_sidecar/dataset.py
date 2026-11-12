"""Small, deduplicated local decision screenshots with explicitly weak labels."""
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import threading


class DecisionRecorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.seen = set()
        self.predicted = set()

    def append(self, event):
        with self.lock, (self.directory / 'samples.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(), **event}, ensure_ascii=False) + '\n')

    def record(self, image, labels):
        # River OCR jitter must not create many copies of one decision image.
        identity = {k: labels[k] for k in ('hand', 'screen') if k in labels} or labels
        serialized = json.dumps(identity, ensure_ascii=False, sort_keys=True)
        key = hashlib.sha256(serialized.encode()).hexdigest()[:20]
        with self.lock:
            if key in self.seen:
                return key
            self.seen.add(key)
        image = image.convert('RGB')
        image.thumbnail((1280, 1280))
        for quality in (78, 68, 58, 45):
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=quality, optimize=True)
            if buffer.tell() <= 160 * 1024:
                break
        while buffer.tell() > 160 * 1024:
            image.thumbnail((int(image.width * .85), int(image.height * .85)))
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=45, optimize=True)
        filename = key + '.jpg'
        target = self.directory / filename
        if not target.exists():
            target.write_bytes(buffer.getvalue())
            self.append({'event': 'sample', 'id': key, 'image': filename, 'bytes': target.stat().st_size,
                         'size': list(image.size), 'labels': labels, 'label_status': 'unreviewed_model_output'})
        return key

    def prediction(self, sample_id, result, latency):
        # One image can be reused after public state/manual metadata changes.
        # Deduplicate actual inference inputs, rather than losing the new trace.
        model_input = result.get('model_input')
        identity = hashlib.sha256(json.dumps(model_input, sort_keys=True).encode()).hexdigest() if model_input else None
        key = (sample_id, identity)
        with self.lock:
            if key in self.predicted:
                return
            self.predicted.add(key)
        self.append({'event': 'prediction', 'id': sample_id, 'latency': latency, 'result': result})
