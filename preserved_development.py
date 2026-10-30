"""Exploratory incremental gains against the preserved policy on identical dev walls."""
from scripts.audit_score_confirmation import compare


def compare_preserved(baseline,anchor,variants,config):
    audit_config={**config,'confirmation_seed_start':config['development_seed_start'],
                  'confirmation_seeds':config['development_seeds']}
    old=compare(anchor,baseline,audit_config)
    reports=[]
    for variant in variants:
        computed=compare(variant['rows'],anchor,audit_config)
        reports.append({'variant':variant['variant'],'checkpoint':variant['checkpoint'],
                        'metrics':computed['candidate'],'gain_vs_preserved':computed['primary_delta'],
                        'paired_ci95_vs_preserved':computed['primary_delta_ci95'],
                        'changed_outcomes':computed['changed_outcomes']})
    return {'scope':'exploratory development comparison, not independent improvement confirmation',
            'preserved_metrics':old['candidate'],'original_metrics':old['reference'],
            'preserved_gain_vs_original':old['primary_delta'],
            'preserved_vs_original_ci95':old['primary_delta_ci95'],'variants':reports,
            'main_selection_changed':False,'primary_confirmation_rule_changed':False}
