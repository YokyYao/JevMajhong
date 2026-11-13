"""Explicit same-component continuation from a pinned calling context."""
from jevmahjong.policy.call_baseline import load


def initial_context(config, model_files):
    mode=config.get('context_initialization','zero_output')
    calling=config.get('train_scopes')==['calling'] and config.get('decision_filter')=='all'
    pinned=calling and bool(config.get('call_context_baseline'))
    if mode not in ('zero_output','calling_baseline'):
        raise ValueError('Unknown context initialization')
    if pinned != (mode=='calling_baseline'):
        raise ValueError('Pure-call baseline data requires explicit calling_baseline initialization')
    if mode=='zero_output':return None
    context=load(config,model_files)
    if (context is None or context.get('model_kind')!='mlp'
            or context['version']!=config['feature_version']
            or context['hidden']!=config['hidden']
            or context['reference_prior_strength']!=config['reference_prior_strength']):
        raise ValueError('Continuation must preserve the calling context schema and prior strength')
    return context
