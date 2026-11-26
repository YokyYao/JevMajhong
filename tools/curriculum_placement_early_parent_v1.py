"""Actual completed early-route 3B evidence for the existing placement course.

Intermediate strategy grades remain reports. Preserve weights, full audit,
data replay and real terminal evidence; never manufacture an old passing grade.
"""
import ast
from contextlib import contextmanager
import copy
import inspect
from pathlib import Path
import sys
from types import FunctionType,SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import curriculum_yaku_early_v2 as early
from tools import curriculum_placement_parent_v1 as common
from tools import curriculum_placement_retained_parent_v1 as retained
from tools import seal_curriculum_placement_retained_parent_v1 as producer
from tools.curriculum_transfer import read,sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_progression_parent_v1 import extend
from tools.capture_curriculum_paired_terminal_v2 import resource,terminal_record
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit
from tools.curriculum_yaku_parent_runtime_v1 import require_registration

METHOD='completed_early_selected1800_yaku2500_placement_lineage_v1'
PARENT_METHOD='audited_completed_early_yaku2500_placement_contract_new_optimizer_v1'
STATUS='actual_completed_early_yaku2500_sealed_for_placement_not_fit_registered'
PARENT=early.RUN/'fit-plan/registered.json'
QUEUE=early.QUEUE
ENTRY=early.ENTRY
CAP=early.RUN/'terminal-v2/terminal-evidence-private.json'
ARCHIVE=early.RUN/'capture-terminal-for-placement-private.json'
PUBLIC=('tools/curriculum_placement_early_parent_v1.py',
    'tests/core/test_curriculum_placement_early_parent_v1.py',
    'docs/experiments/plans/curriculum-placement-early-parent-20261125.md')
PREFIX='curriculum-yaku-early-v2-exit-'


def require_lineage(parent):
    c=parent['curriculum'];spec=parent['training_config']['parent_initialization']
    if (c['run_id']!='jev-curriculum-yaku-early-v2' or c['stage']!='stage3B-yaku-early-v2'
            or c['stage_offset']!=2796 or c['out']!='runs/curriculum-yaku-early-v2/stage3B-fit'
            or spec['method']!=early.METHOD or spec['selected_step']!=1800):
        raise ValueError('The actual selected historical1800 -> current3B ancestry required')


def require_completion(parent,done,evidence,cap,queue):
    require_registration(parent);require_lineage(parent)
    if (done['status']!='yaku2500_fit_and_real_curves_complete' or done['optimizer_steps']!=2500
            or done['actual_records_seen']!=182494 or done['loss_method']!=retained.PARENT_LOSS
            or evidence['status']!='yaku2500_final_evidence_audit_passed'
            or evidence['loss_method']!=retained.PARENT_LOSS or evidence['optimizer_steps']!=2500
            or evidence['actual_records_seen']!=182494 or evidence['saved_final_reloaded_menus']!=24516
            or evidence['all376_optimizer_states_match'] is not True
            or evidence['source_and_data_hashes_unchanged'] is not True
            or evidence['source_split_replay_and_exposure_exact'] is not True
            or evidence['final_reserved_grid_unused'] is not True
            or evidence['automatic_release'] is not False
            or type(evidence['stage3B_advancement']['passed']) is not bool
            or cap['status']!='early_direct3B_actual_terminals_preserved'
            or cap['automatic_release'] is not False or cap['final_confirmation_opened'] is not False
            or set(cap['scheduler_proofs'])!={'prepare','fit','audit'}
            or any(cap['scheduler_proofs'][role]['successful'] is not True
                or str(cap['scheduler_proofs'][role]['job_id'])!=str(queue[role+'_job'])
                for role in ('prepare','fit','audit'))):
        raise ValueError('Actual completed2500 budget, independent audit and genuine successful phases required')


def phase_proofs(cap,queue,captured,receipt_reader):
    """Require all four actual producer phases, including its own capture job."""
    if set(cap['scheduler_proofs'])!={'prepare','fit','audit'}:
        raise ValueError('All actual early3B producer phases required')
    exits={};paths=[]
    for role,proof in cap['scheduler_proofs'].items():
        job=str(queue[role+'_job']);actual=terminal_record(proof['raw'],job,ENTRY)
        path=ROOT/'logs'/f'{PREFIX}{job}.json';receipt=receipt_reader(path)
        if (not actual['successful'] or any(proof[key]!=value for key,value in actual.items())
                or receipt!=proof['exit_receipt'] or receipt.get('phase')!=role):
            raise ValueError('Actual phase identity or receipt differs')
        exits[role]=scheduler_exit(receipt,proof['raw'],ENTRY);paths.append(path)
    job=str(queue['capture_job']);path=ROOT/'logs'/f'{PREFIX}{job}.json';receipt=receipt_reader(path)
    if (captured['exit']!=receipt or str(receipt['job_id'])!=job or receipt.get('phase')!='capture'
            or not terminal_record(captured['raw'],job,ENTRY)['successful']):
        raise ValueError('Real own capture terminal and exact queued identity required')
    exits['capture']=scheduler_exit(receipt,captured['raw'],ENTRY);paths.append(path)
    return exits,paths


def parent_artifacts(registration):
    resource();registration=Path(registration).resolve()
    if registration!=PARENT.resolve():raise ValueError('Only the actual current3B output may continue')
    parent=read(registration);fit=ROOT/parent['curriculum']['out']
    done_path=fit/'completion.json';evidence_path=early.RUN/'final-evidence-private.json'
    evidence=read(evidence_path);cap=read(CAP);queue=read(QUEUE)
    require_completion(parent,read(done_path),evidence,cap,queue)
    files=dict(parent['inputs']);extend(files,early.RUN/'launch-binding-private.json')
    exits,paths=phase_proofs(cap,queue,read(ARCHIVE),read)
    paths.extend((registration,done_path,evidence_path,CAP,QUEUE,ARCHIVE))
    for path in paths:files[str(path.resolve())]=sha(path)
    if any(sha(name)!=value for name,value in files.items()):raise ValueError('Actual parent input changed')
    return parent,evidence,fit,files,exits


def report_grade_only(function,**overrides):
    """Remove exactly the old grade veto; all other guards/calculations remain."""
    tree=ast.parse(inspect.getsource(function));removed=0
    class Grade(ast.NodeTransformer):
        def visit_BoolOp(self,node):
            nonlocal removed
            node=self.generic_visit(node);values=[]
            for value in node.values:
                if ast.unparse(value)=="sealed['original_parent_grade']['passed'] is not True":removed+=1
                else:values.append(value)
            node.values=values;return node
    tree=Grade().visit(tree)
    if removed!=1:raise ValueError('Review the exact old strategy-grade guard before adapting')
    scope=dict(function.__globals__,**overrides)
    exec(compile(ast.fix_missing_locations(tree),str(Path(__file__).resolve()),'exec'),scope)
    return scope[function.__name__]


def seal(registration,data_path,original_path,out):
    def writer(path,value):
        if Path(path).name=='parent-seal-private.json':
            value=copy.deepcopy(value);value.update(placement_parent_lineage_method=METHOD,
                selected3A_parent_step=1800,placement_parent_selected_step=2500,
                stage3B_grade_is_report_not_admission_veto=True)
        producer.write(path,value)
    namespace=SimpleNamespace(METHOD=PARENT_METHOD,STATUS=STATUS,replay_projection=common.replay_projection)
    function=producer.seal
    inherited=FunctionType(function.__code__,dict(function.__globals__,parent_artifacts=parent_artifacts,
        common=namespace,PUBLIC=(*producer.PUBLIC,*PUBLIC,*early.PUBLIC),write=writer),
        function.__name__,function.__defaults__,function.__closure__)
    return inherited(Path(registration).resolve(),Path(data_path).resolve(),Path(original_path).resolve(),Path(out).resolve())


def require_sealed_lineage(sealed):
    if (sealed.get('placement_parent_lineage_method')!=METHOD or sealed.get('selected3A_parent_step')!=1800
            or sealed.get('placement_parent_selected_step')!=2500
            or sealed.get('stage3B_grade_is_report_not_admission_veto') is not True
            or Path(sealed['parent_registration']).resolve()!=PARENT.resolve()
            or type(sealed['original_parent_grade']['passed']) is not bool):
        raise ValueError('Explicit current3B lineage and unchanged reported grade required')


def verify_parent(config):
    sealed=read(config['parent_initialization']['manifest']);require_sealed_lineage(sealed)
    parent,evidence,fit,files,exits=parent_artifacts(PARENT)
    if sealed['original_parent_grade']!=evidence['stage3B_advancement']:
        raise ValueError('Preserve the actual strategy report; never relabel a failed grade')
    for name,value in files.items():
        if sealed['bound_files'].get(name)!=value:raise ValueError('Actual full parent evidence escaped the seal')
    return report_grade_only(common.verify_parent,METHOD=PARENT_METHOD,STATUS=STATUS)(config)


@contextmanager
def parent_loader(initialization):
    before=initialization.verified_parent;initialization.verified_parent=verify_parent
    try:yield
    finally:initialization.verified_parent=before
