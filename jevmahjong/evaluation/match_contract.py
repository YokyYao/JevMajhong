"""Bind a calling comparison to its declared candidate and reference policy.

Only standard-library dependencies are used so independent record auditors can
load this checker separately from the evaluator's frozen package.
"""
import hashlib
from pathlib import Path


def require(condition,message):
    if not condition:raise ValueError(message)


def calling_comparison_contract(config,protocol,fit_protocol,model,
                                reference_baseline_sha256,candidate_activation):
    """Validate identities; event recounting still has to prove game results."""
    require(protocol['config']==config,'Evaluation startup configuration differs')
    require(config.get('confirmation_objective')=='hand_win'
        and config.get('updated_scopes')==['calling'],'Calling hand-win comparison required')
    require(candidate_activation in ('all','call_windows')
        and config.get('candidate_activation')==candidate_activation,'Prescribed candidate activation differs')
    require(model.get('base_and_LoRA_frozen') is True
        and model.get('trained_scopes')==['calling']
        and fit_protocol['config'].get('train_scopes')==['calling'],'Candidate scope or frozen encoders differ')
    require(protocol['model_files']==fit_protocol['model_files']==model['model_files'],
        'Comparison encoder identities differ')
    declaration=config.get('call_context_baseline')
    if reference_baseline_sha256 is None:
        require(declaration is None,'Unexpected stronger reference policy')
        baseline=None
    else:
        require(isinstance(declaration,dict) and set(declaration)=={'checkpoint','sha256'},
            'Explicit stronger reference checkpoint required')
        require(declaration['sha256']==reference_baseline_sha256,'Prescribed reference policy differs')
        path=Path(declaration['checkpoint']).resolve()
        require(hashlib.sha256(path.read_bytes()).hexdigest()==reference_baseline_sha256,
            'Reference checkpoint changed')
        baseline={'checkpoint':str(path),'sha256':reference_baseline_sha256}
    require(all(p.get('call_context_baseline')==baseline for p in (protocol,fit_protocol,model)),
        'Candidate, fit and evaluation reference policies differ')
    replaces=model.get('context_replaces_calling_baseline',False)
    require(isinstance(replaces,bool),'Explicit context replacement flag required')
    if model.get('kan_context') is not None:
        require(candidate_activation=='call_windows' and model.get('decision_filter')=='call_windows',
            'Context candidate must use its calling-only activation')
        require(replaces==(baseline is not None),'Calling context would omit or duplicate the fixed baseline')
        if replaces:require(model.get('context_initialization')=='calling_baseline',
            'Calling-context continuation identity missing')
        component='calling_context'
        parity={'ordinary':2,'calling':2,'ordinary_kan':2}
    else:
        require(candidate_activation=='all' and model.get('decision_filter')=='all' and not replaces,
            'Pointer candidate must retain its fixed context rather than replace it')
        component='calling_pointer_head'
        parity={'ordinary':2,'calling':2}
    return {'candidate_component':component,'reference_baseline_sha256':reference_baseline_sha256,
        'candidate_activation':candidate_activation,'expected_native_parity':parity}


def check_evaluator_package(protocol,source):
    """The declared package must be the exact package recorded at startup."""
    package=(Path(source)/'jevmahjong').resolve();files=protocol.get('code_sha256',{})
    require(bool(files),'Evaluator source receipt missing')
    for name,checksum in files.items():
        path=Path(name).resolve()
        require(path.is_relative_to(package),'Evaluator package differs from startup source')
        require(hashlib.sha256(path.read_bytes()).hexdigest()==checksum,'Evaluator source changed')
