"""Bind a kan-only candidate while preserving the verified calling baseline."""
import hashlib
from pathlib import Path
from jevmahjong.evaluation.match_contract import check_evaluator_package


def require(condition, message):
    if not condition:
        raise ValueError(message)


def kan_comparison_contract(config, protocol, fit_protocol, model,
                            reference_baseline_sha256, candidate_activation):
    require(protocol['config'] == config, 'Evaluation startup configuration differs')
    require(config.get('confirmation_objective') == 'hand_win'
            and config.get('updated_scopes') == ['ordinary'], 'Kan hand-win comparison required')
    require(candidate_activation == config.get('candidate_activation') == 'kan_windows',
            'Prescribed kan activation differs')
    require(config.get('train_scopes') == ['ordinary']
            and config.get('decision_filter') == 'kan_windows', 'Explicit kan workflow required')
    fit = fit_protocol['config']
    require(model.get('base_and_LoRA_frozen') is True
            and model.get('trained_scopes') == fit.get('train_scopes') == ['ordinary']
            and model.get('decision_filter') == fit.get('decision_filter') == 'kan_windows',
            'Candidate scope or frozen encoders differ')
    require(protocol['model_files'] == fit_protocol['model_files'] == model['model_files'],
            'Comparison encoder identities differ')
    require(model.get('kan_context') is not None, 'Kan context candidate required')
    require(model.get('context_replaces_calling_baseline', False) is False,
            'Kan candidate must preserve the single calling baseline')
    declaration = config.get('call_context_baseline')
    require(reference_baseline_sha256 is not None and isinstance(declaration, dict)
            and set(declaration) == {'checkpoint', 'sha256'}, 'Explicit stronger reference required')
    require(declaration['sha256'] == reference_baseline_sha256, 'Prescribed reference policy differs')
    path = Path(declaration['checkpoint']).resolve()
    require(hashlib.sha256(path.read_bytes()).hexdigest() == reference_baseline_sha256,
            'Reference checkpoint changed')
    baseline = {'checkpoint': str(path), 'sha256': reference_baseline_sha256}
    require(all(p.get('call_context_baseline') == baseline
                for p in (protocol, fit_protocol, model)),
            'Candidate, fit and evaluation reference policies differ')
    return {'candidate_component': 'kan_context',
            'reference_baseline_sha256': reference_baseline_sha256,
            'candidate_activation': 'kan_windows',
            'expected_native_parity': {'ordinary': 2, 'calling': 2, 'ordinary_nonkan': 2}}
