"""Restore the exact public training packages used by curriculum stage1."""
import hashlib
import io
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT=Path(__file__).resolve().parents[1]
RESEARCH_REV='2f0c84a3f9c813990da17ef7dc4ba13f7b68979e'
JEVANY_REV='c740121a4532592b09b6272104d5f29a94c232e5'
SOURCE_SHA='7a6e9471c95d529ecdc5d0a66cd240ae17d1f5914d76ce8dbd61ad45ce7104bc'
UPSTREAM_SHA='84fe7f0201a8d9928aabe4770912bf25e76c4916c1eded3011cf8f02862b758d'

def digest_tree(directory):
    value=hashlib.sha256()
    for p in sorted(directory.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc':continue
        value.update(p.relative_to(directory).as_posix().encode());value.update(b'\0')
        value.update(hashlib.sha256(p.read_bytes()).digest())
    return value.hexdigest()

def package_from_archive(repo,revision,package,destination):
    if destination.exists():raise FileExistsError(destination)
    kind='research' if package=='jevmahjong' else 'upstream'
    files=json.loads((ROOT/'configs/environment/curriculum-code-files.json').read_text())[kind]
    destination.mkdir(parents=True)
    def fetch(spec):
        relative=PurePosixPath(spec['path'])
        if relative.parts[0]!=package or '..' in relative.parts:raise ValueError('Unsafe source path')
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/{repo}/{revision}/{spec["path"]}',timeout=45) as response:
            content=response.read()
        actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if actual!=spec['sha']:raise ValueError('Public source Git blob mismatch')
        target=destination.joinpath(*relative.parts[1:]);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    # Fetch only code files, never the repository's unrelated data/weight archives.
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fetch,files))

def bootstrap():
    assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS'),'Scheduled GPU required'
    source=ROOT/'runs/code'/SOURCE_SHA;upstream=ROOT/'runs/upstream-code'/UPSTREAM_SHA
    if not source.exists():
        package_from_archive('YokyYao/JevMajhong',RESEARCH_REV,'jevmahjong',source/'jevmahjong')
        shutil.copy2(ROOT/'reproduction/overlays/training/initialization.py',source/'jevmahjong/training/initialization.py')
        shutil.copy2(ROOT/'jevmahjong/training/metric_journal.py',source/'jevmahjong/training/metric_journal.py')
    if digest_tree(source/'jevmahjong')!=SOURCE_SHA:raise ValueError('Training package differs from registered run')
    if not upstream.exists():
        package_from_archive('SimpleJev/JevAny',JEVANY_REV,'jevany',upstream/'jevany')
        for notice in json.loads((ROOT/'configs/environment/curriculum-code-files.json').read_text())['notices']:
            with urllib.request.urlopen(f'https://raw.githubusercontent.com/SimpleJev/JevAny/{JEVANY_REV}/{notice["path"]}',timeout=45) as response:
                content=response.read()
            actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
            if actual!=notice['sha']:raise ValueError('Upstream license/notice changed')
            (upstream/notice['path']).write_bytes(content)
    if digest_tree(upstream/'jevany')!=UPSTREAM_SHA:raise ValueError('JevAny package differs from registered run')
    report=dict(training_source=str(source),training_source_sha256=SOURCE_SHA,
        upstream_source=str(upstream),upstream_source_sha256=UPSTREAM_SHA,
        research_revision=RESEARCH_REV,jevany_revision=JEVANY_REV)
    path=ROOT/'runs/curriculum-reproduction-v1/bootstrap.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':bootstrap()
