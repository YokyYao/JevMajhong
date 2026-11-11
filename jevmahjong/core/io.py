import hashlib
import json
from pathlib import Path

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def request_hash(request):
    value = json.loads(json.dumps(request))
    value.pop('_meta', None)
    value['questions']['action'].pop('label', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)
