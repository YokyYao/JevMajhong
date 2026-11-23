"""Read-only inherited split-field inventory; execute only on scheduled GPU."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha


def main():
    gpu_only(); out = ROOT/'runs/curriculum-progression-union-v1/partition-diagnosis-v1'
    out.mkdir(parents=True, exist_ok=False)
    manifests = dict(calls='runs/curriculum-calls-post-reach-data-v1/audit-v1/summary-private.json',
        risk='runs/curriculum-risk-teacher-data-v1/audit-v1/summary-private.json',
        ordinary='runs/curriculum-ordinary-completion-v1/audit-v1/summary-private.json',
        old_mc='runs/curriculum-stage3-attack-v1/data/data-ready.json')
    result = {}; examples = []; files = {str(Path(__file__)): sha(__file__)}
    for cohort, name in manifests.items():
        path = ROOT/name; manifest = read(path); files[str(path)] = sha(path)
        for split, spec in manifest['files'].items():
            data = ROOT/spec['path']
            if sha(data) != spec['sha256']: raise ValueError('Bound source data changed')
            files[str(data)] = spec['sha256']; counts = Counter()
            for line in data.open():
                row = json.loads(line); m = row['_meta']
                if cohort == 'old_mc' and m['label_type'] != 'paired_future_masked': continue
                field = m.get('split', '<missing>'); teacher = 'source_logical_seed' in m
                counts['rows'] += 1; counts['field:'+str(field)] += 1
                counts['Mortal_source_logical_seed_present'] += teacher
                if field != split and field != '<missing>':
                    counts['mismatched_supplied_field'] += 1
                    if len([e for e in examples if e['cohort'] == cohort and e['file_split'] == split]) < 3:
                        examples.append(dict(cohort=cohort, file_split=split, id=m['id'], group=m['group_id'],
                            supplied_split=field, label_type=m.get('label_type'), keys=sorted(m), teacher=teacher))
            result[cohort+':'+split] = dict(counts)
    if any(sha(p) != h for p, h in files.items()): raise ValueError('Immutable source changed')
    write(out/'examples-private.json', examples); write(out/'binding-private.json', dict(files=files))
    write(out/'summary-private.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
