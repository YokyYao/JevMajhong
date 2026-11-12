"""Run the maintained CLI from a private immutable source snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def tree_digest(package):
    value = hashlib.sha256()
    for path in sorted(package.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc': continue
        value.update(path.relative_to(package).as_posix().encode()); value.update(b'\0')
        value.update(hashlib.sha256(path.read_bytes()).digest())
    return value.hexdigest()


def freeze_source(root):
    package = root / 'jevmahjong'; identity = tree_digest(package)
    target = root / 'runs/code' / identity
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=target.parent, prefix='preparing-') as temp:
            stage = Path(temp) / 'source'; stage.mkdir()
            shutil.copytree(package, stage / 'jevmahjong', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            if tree_digest(stage / 'jevmahjong') != identity: raise ValueError('Source changed during snapshot')
            try: stage.rename(target)
            except FileExistsError: pass
    if tree_digest(target / 'jevmahjong') != identity: raise ValueError('Saved source snapshot changed')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('collect', 'collect-defense-states', 'collect-ev', 'audit-defense-states', 'audit-futures', 'audit-context-fit', 'audit-call-head', 'train', 'train-context', 'train-call-head', 'evaluate'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Pin source/configuration without starting work')
    parser.add_argument('--registered', action='store_true', help='Use a previously pinned launch; do not resume module state')
    parser.add_argument('--max-walls',type=int)
    args = parser.parse_args(); config = json.loads(args.config.read_text())
    if args.max_walls is not None and args.command!='collect-defense-states':
        raise ValueError('Bounded prefix collection is only supported for native defense states')
    out = (ROOT / config['out']).resolve()
    key = hashlib.sha256(str(out).encode()).hexdigest()
    launches = ROOT / 'runs/launches'; launches.mkdir(parents=True, exist_ok=True)
    record = launches / (key + '.json')
    if args.registered and not record.exists():
        raise ValueError('Register this exact command/configuration with --prepare-only before submitting')
    if record.exists():
        saved = json.loads(record.read_text())
        if not (args.resume or args.registered or args.prepare_only) or saved['config'] != config or saved['command'] != args.command:
            raise ValueError('Use --resume with the same configuration, or choose a new output directory')
        source = Path(saved['source'])
        if tree_digest(source / 'jevmahjong') != saved['source_sha256']:
            raise ValueError('Original source snapshot changed')
    else:
        source = freeze_source(ROOT)
        saved = {'command': args.command, 'source': str(source), 'source_sha256': source.name, 'config': config}
        record.write_text(json.dumps(saved, indent=2) + '\n')
    frozen_config = launches / (key + '-config.json')
    if frozen_config.exists() and json.loads(frozen_config.read_text()) != config:
        raise ValueError('Saved launch configuration changed')
    frozen_config.write_text(json.dumps(config, indent=2) + '\n')
    if args.prepare_only:
        print(json.dumps({'status':'registered_not_started','source_sha256':saved['source_sha256'],
            'launch':str(record),'config':str(frozen_config)}))
        return
    bootstrap = ('import sys;sys.path.insert(0,sys.argv[1]);'
        'from jevmahjong.__main__ import main;sys.argv=[sys.argv[0],*sys.argv[2:]];main()')
    command = [sys.executable, '-c', bootstrap, str(source), args.command, '--config', str(frozen_config)]
    if args.resume: command.append('--resume')
    if args.max_walls is not None: command.extend(['--max-walls',str(args.max_walls)])
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__': main()
