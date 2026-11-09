"""Build a deduplicated, explicitly unreviewed index of a local game run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

from PIL import Image


def audit(directory, events_path):
    records = [json.loads(line) for line in (directory / 'samples.jsonl').read_text(encoding='utf8').splitlines()]
    seen, reviews, selected = set(), [], []
    for record in records:
        if record.get('event') != 'sample':
            continue
        labels = record['labels']
        hand, screen = labels.get('hand', []), labels.get('screen', {})
        identity = json.dumps({'hand': hand, 'screen': screen}, sort_keys=True)
        key = hashlib.sha256(identity.encode()).hexdigest()
        path = directory / record['image']
        reason = None
        with Image.open(path) as image:
            width, height = image.size
        if width < 400 or height < 200:
            reason = 'synthetic_or_small_fixture'
        elif screen.get('reaction') or screen.get('offered'):
            reason = 'reaction_target_requires_visual_review'
        elif len(hand) not in (2, 5, 8, 11, 14):
            reason = 'nondecision_or_animation_frame'
        elif screen.get('melds') is None or len(screen['melds']) != (14 - len(hand)) // 3:
            reason = 'unresolved_meld_state'
        elif key in seen:
            reason = 'duplicate_hand_and_action_state'
        seen.add(key)
        reviews.append({'id': record['id'], 'image': record['image'],
                        'status': 'excluded' if reason else 'unreviewed_game_capture',
                        'reason': reason, 'policy_label': 'model_output_not_ground_truth'})
        if not reason:
            selected.append(record)
    for filename, values in [('review.jsonl', reviews), ('unreviewed-index.jsonl', selected)]:
        (directory / filename).write_text(''.join(json.dumps(v, ensure_ascii=False) + '\n' for v in values), encoding='utf8')
    events = [json.loads(line) for line in events_path.read_text(encoding='utf8').splitlines()]
    latency = sorted(v['latency'] for v in events if v.get('event') == 'recommendation'
                     and v.get('engine') == 'Jev 校正版2' and isinstance(v.get('latency'), (int, float)))
    images = list(directory.glob('*.jpg'))
    for path in images:
        with Image.open(path) as image:
            if max(image.size) > 1280 or path.stat().st_size > 160 * 1024:
                raise ValueError(f'Capture exceeds compression limits: {path}')
    summary = {'images': len(images), 'total_MiB': round(sum(p.stat().st_size for p in images) / 2**20, 2),
               'max_KiB': round(max((p.stat().st_size for p in images), default=0) / 1024, 2),
               'unreviewed_unique_decisions': len(selected), 'excluded_samples': len(reviews) - len(selected),
               'exclusion_reasons': dict(Counter(r['reason'] for r in reviews if r['reason'])),
               'max_dimension': 1280, 'trained_weights': False,
               'jev_screenshot_to_advice_seconds': {'count': len(latency), 'median': round(statistics.median(latency), 3) if latency else None,
                   'p95': round(latency[min(len(latency)-1, int(len(latency)*.95))], 3) if latency else None,
                   'max': round(max(latency), 3) if latency else None, 'within_3s': sum(v <= 3 for v in latency)},
               'note': 'All sessions including debugging; not a win-rate or accuracy evaluation.'}
    (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    archive = directory / 'unreviewed-decisions.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for record in selected:
            bundle.write(directory / record['image'], record['image'])
        for filename in ('unreviewed-index.jsonl', 'summary.json', 'README.txt'):
            bundle.write(directory / filename, filename)
    print(f'Unreviewed archive: {archive.stat().st_size / 2**20:.2f} MiB')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('events', type=Path)
    args = parser.parse_args()
    audit(args.directory, args.events)
