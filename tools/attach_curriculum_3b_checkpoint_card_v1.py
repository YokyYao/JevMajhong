"""Create a new paper-material version with the actual usable checkpoint card."""
import argparse
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write


def attach(material, checkpoint, out):
    allocation = resource()
    material, checkpoint, out = (Path(p).resolve() for p in (material, checkpoint, out))
    if out.exists() or not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Use a new paper-material version')
    source_binding = material/'completion-binding-private.json'; bound = read(source_binding)['files']
    checkpoint_binding = checkpoint/'completion-binding-private.json'
    cp_bound = read(checkpoint_binding)['files']
    cp_path = checkpoint/'inference-manifest-private.json'
    if cp_bound.get(str(cp_path)) != sha(cp_path): raise ValueError('Actual checkpoint export changed')
    cp = read(cp_path); summary = read(material/'summary-private.json')
    if (cp['status'] != 'actual_evaluated_adapter_head_and_runtime_exported_and_load_query_passed'
            or not cp['packaged_load_and_query_passed'] or cp['local_step'] != 2500
            or cp['final2500_audited'] is not True or summary['last_step'] != 2500
            or not summary['actual_final2500_audited']
            or summary['evaluated_parameter_identity'] != cp['parameter_identity']):
        raise ValueError('The paper must describe the genuinely audited and usable same2500 model')
    out.mkdir(parents=True)
    for source in material.rglob('*'):
        if not source.is_file() or source.name in ('completion-binding-private.json', 'summary-private.json'): continue
        if bound.get(str(source)) != sha(source): raise ValueError('Completed material changed')
        target = out/source.relative_to(material); target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    identity = cp['parameter_identity']
    card = r'''
\paragraph{Usable audited checkpoint.}
The prepared research artifact is \textbf{Jev Curriculum 3B, step 2,500}, selected
by the registered endpoint budget rather than minimum validation loss.
Its adapter, decision head and tokenizer were copied byte-for-byte; the packaged
frozen runtime loaded that copy and successfully scored an actual non-reserved
validation request. The original training checkpoint, including optimizer state,
is retained separately. The inference bundle requires the pinned Qwen base
specified above and includes a request CLI and recorded environment versions.
Its full parameter identity is
\texttt{FIRSTHALF}\newline
\texttt{SECONDHALF}.
The evaluated native policy scores the complete legal menu, applies the fixed
RI0.5 rule to reach/discard menus, and retains available-win and single-action
shortcuts. These execution rules are part of the reported policy, not evidence
that every action path is learned. The artifact is a research checkpoint;
independent frozen-v8 confirmation remains pending.
'''.replace('FIRSTHALF', identity[:32]).replace('SECONDHALF', identity[32:])
    section = out/'curriculum_reproduction.tex'; section.write_text(section.read_text()+card)
    summary.update(usable_checkpoint_parameter_identity=identity, actual_packaged_load_query_passed=True,
        checkpoint_card_attached=True, allocation=allocation)
    write(out/'summary-private.json', summary)
    files = {str(source_binding):sha(source_binding), str(checkpoint_binding):sha(checkpoint_binding),
        str(cp_path):sha(cp_path), str(Path(__file__)):sha(__file__)}
    for path in out.rglob('*'):
        if path.is_file(): files[str(path)] = sha(path)
    write(out/'completion-binding-private.json', dict(files=files))
    print(dict(status='actual_usable2500_checkpoint_card_attached', parameter_identity=identity), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); attach(args.material, args.checkpoint, args.out)
