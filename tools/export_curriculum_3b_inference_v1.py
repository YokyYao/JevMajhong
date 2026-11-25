"""Export a complete adapter/head bundle, pinned runtime and usable request CLI.

Private artifacts only. A2500 export requires the genuine completed training,
full reload audit and phase terminals. The copied bundle is loaded and queried
once on a non-reserved validation example; no new games or updates are run.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write

REGISTRATION = ROOT/'runs/curriculum-yaku-early-v2/fit-plan/registered.json'
INFER = r'''"""Run in a scheduler GPU job with the project's existing Python environment."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'runtime'))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--request', type=Path, required=True)
a = p.parse_args()
job = os.environ.get('SLURM_JOB_ID', '')
observed = subprocess.run(['scontrol', '-o', 'show', 'job', job], capture_output=True, text=True, timeout=20)
fields = dict(t.split('=', 1) for t in observed.stdout.split() if '=' in t)
tres = dict(t.split('=', 1) for t in fields.get('AllocTRES', '').split(',') if '=' in t)
if (not job.isdigit() or observed.returncode or fields.get('JobState') != 'RUNNING'
        or int(tres.get('gres/gpu', '0')) < 1 or socket.gethostname().startswith(('lo-', 'login'))
        or fields.get('NodeList') != socket.gethostname().split('.')[0]):
    raise RuntimeError('An actual finite allocated GPU node is required before loading the model')
from jevmahjong.policy.routing import load_model
from jevmahjong.policy.unified import UnifiedPolicyClient
from tools.curriculum_complete_legal_class_policy_v1 import select
from jevany.api import validate_distribution
request = json.loads(a.request.read_text())
criteria = request['questions']['action']['criteria']; keys = list(criteria)
if not keys: raise ValueError('Provide a complete legal action menu')
kinds = [criteria[key]['type'] for key in keys]
if len(keys) == 1:
    index = 0; probabilities = {keys[0]: 1.}; loaded = False
elif any(kind in ('ron', 'tsumo') for kind in kinds):
    index = next(i for i,k in enumerate(kinds) if k in ('ron', 'tsumo'))
    probabilities = None; loaded = False
else:
    client = UnifiedPolicyClient(load_model(ROOT/'checkpoint'))
    probabilities = dict(client.system_one(**request)['answers']['action']['probabilities'])
    validate_distribution(dict(probabilities), keys)
    index, _ = select(kinds, [probabilities[key] for key in keys]); loaded = True
print(json.dumps(dict(chosen_key=keys[index], action=criteria[keys[index]],
    probabilities=probabilities, model_loaded=loaded,
    selection_method='complete_native_legal_menu_with_fixed_full_class_RI0_5_v1',
    new_games=0, optimizer_updates=0)), flush=True)
'''


def export(step, out):
    allocation = resource(); out = Path(out).resolve()
    if step not in (1875, 2500) or out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a new private1875/2500 inference export')
    reg = read(REGISTRATION); fit = ROOT/reg['curriculum']['out']
    if step == 2500:
        from tools.curriculum_placement_early_parent_v1 import parent_artifacts
        parent_artifacts(REGISTRATION)
    expected_games = 64 if step == 2500 else 16
    evaluation = fit/'evaluations'/f'step-{step:06d}'
    if read(evaluation/'status.json') != dict(status='complete', games=expected_games):
        raise ValueError('A fully evaluated actual saved checkpoint is required')
    sys.path[:0] = [reg['source'], reg['training_config']['upstream_source']]
    from jevmahjong.evaluation.unified_contract import bundle, MODEL_FILES
    from jevmahjong.training.base_lora import package_digest
    from jevany.data import api_request
    from jevmahjong.data.policy_data import policy_request
    source = fit/'training-run/checkpoints'/f'step-{step:06d}'
    original = bundle(source); protocol = read(evaluation/'protocol.json')
    if original != protocol['checkpoint']:
        raise ValueError('Only the exact actually evaluated parameter bundle is eligible')
    ready = read(ROOT/reg['curriculum']['data_ready'])
    validation = ROOT/ready['files']['validation']['path']
    if sha(validation) != ready['files']['validation']['sha256']:
        raise ValueError('Frozen non-reserved validation source changed')
    example = None
    with validation.open() as stream:
        for line in stream:
            row = json.loads(line); q = row['questions']['action']
            types = {value['type'] for value in q['criteria'].values()}
            if len(q['criteria']) > 1 and types <= {'dahai', 'reach'}:
                example = api_request(policy_request(dict(state=row['state'], questions=row['questions'])))
                break
    if example is None: raise ValueError('One actual non-shortcut validation request required')
    out.mkdir(parents=True); checkpoint = out/'checkpoint'; checkpoint.mkdir()
    for name in MODEL_FILES: shutil.copyfile(source/name, checkpoint/name)
    copied = bundle(checkpoint)
    if copied['files'] != original['files'] or copied['identity_sha256'] != original['identity_sha256']:
        raise ValueError('Copied inference parameters/tokenizer differ from the original')
    runtime = out/'runtime'; runtime.mkdir()
    for package, parent in [('jevmahjong', Path(reg['source'])),
                            ('jevany', Path(reg['training_config']['upstream_source']))]:
        shutil.copytree(parent/package, runtime/package,
            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache', '.git'))
        if package_digest(runtime/package) != package_digest(parent/package):
            raise ValueError('The packaged runtime source differs from actual inference')
    (runtime/'tools').mkdir(); (runtime/'tools/__init__.py').write_text('')
    policy_files = ('curriculum_complete_legal_class_policy_v1.py', 'curriculum_riichi_class_policy_v1.py')
    for name in policy_files: shutil.copyfile(ROOT/'tools'/name, runtime/'tools'/name)
    (out/'recommend.py').write_text(INFER)
    write(out/'example-request-private.json', example)
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TORCHDYNAMO_DISABLE='1')
    smoke = subprocess.run([sys.executable, str(out/'recommend.py'), '--request',
        str(out/'example-request-private.json')], cwd=out, env=environment,
        capture_output=True, text=True, timeout=180)
    (out/'load-query-smoke-private.log').write_text(smoke.stdout+smoke.stderr)
    if smoke.returncode: raise RuntimeError('Packaged checkpoint load/query failed; inspect retained private log')
    answer = json.loads(smoke.stdout.strip().splitlines()[-1])
    if answer['model_loaded'] is not True or answer['chosen_key'] not in example['questions']['action']['criteria']:
        raise ValueError('The copied model must actually load and choose an available candidate')
    write(out/'load-query-smoke-private.json', answer)
    versions = {name:version(name) for name in ('torch','transformers','peft','riichienv','safetensors')}
    write(out/'environment.json', versions)
    manifest = dict(status='actual_evaluated_adapter_head_and_runtime_exported_and_load_query_passed',
        display_name='Jev Curriculum 3B', local_step=step, final2500_audited=step == 2500,
        base=reg['training_config']['base'], base_revision=reg['training_config']['base_revision'],
        original_training_checkpoint=str(source), inference_checkpoint=str(checkpoint),
        parameter_identity=original['identity_sha256'], policy=protocol['policy'],
        actual_evaluation_games=expected_games, runtime_source_sha256=reg['source_sha256'],
        upstream_source_sha256=reg['training_config']['upstream_source_sha256'],
        packaged_load_and_query_passed=True, new_games=0, optimizer_updates=0,
        raw_weights_private=True, production_replaced=False, allocation=allocation)
    write(out/'inference-manifest-private.json', manifest)
    usage = f'''# Jev Curriculum 3B — step {step}

This private bundle contains the exact evaluated LoRA adapter, action head,
tokenizer and frozen inference source. Parameter identity:
{original['identity_sha256']}

Base: {manifest['base']} at revision {manifest['base_revision']}.
The frozen base weights are required separately; the existing project GPU
environment already caches them. JEV_BASE_MODEL_DIR may identify that exact
snapshot. environment.json records the actual library versions.

Run only inside a finite scheduler GPU allocation using the project's existing
Python environment:

    /home/junchi.yao/mahjong-discard-defense/.venv/bin/python recommend.py --request example-request-private.json

For your own state, supply its complete legal candidates in the same structured
request schema. The packaged CLI uses the actual full-legal-menu RI0.5 selection
rule, including available-win and one-action shortcuts. For native observations,
use runtime/tools/curriculum_complete_legal_class_policy_v1.py's
CompleteLegalClassPolicyAgent with UnifiedPolicyClient(load_model(checkpoint)).
This is a tested research checkpoint, not a claim that an existing EXE uses it.
The source training checkpoint (including optimizer state) remains preserved at:
{source}

The included example is a private validation request. Keep it, this experimental
bundle and all smoke details private. No reserved final games were opened.
'''
    (out/'USAGE.md').write_text(usage)
    files = {str(Path(__file__)): sha(__file__), str(REGISTRATION):sha(REGISTRATION),
        str(evaluation/'protocol.json'):sha(evaluation/'protocol.json')}
    for path in out.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts: files[str(path)] = sha(path)
    write(out/'completion-binding-private.json', dict(files=files))
    print(json.dumps(dict(status=manifest['status'], step=step,
        parameter_identity=original['identity_sha256'], loaded_and_queried=True,
        final2500_audited=step == 2500)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); export(args.step, args.out)
