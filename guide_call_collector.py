"""Frozen counterfactual point collection with public strategy covariates."""
import torch
from call_counterfactual import CallInterventionPolicy
from guide_call_features import VERSION, NAMES, strategy_features
from scoring_audit import settlement_ledger


class GuideCallInterventionPolicy(CallInterventionPolicy):
    def act(self, obs, frozen=False, stochastic=False, trace=None):
        local = [] if trace is not None else None
        chosen = super().act(obs, frozen=frozen, stochastic=stochastic, trace=local)
        if trace is not None:
            for row in local:
                row.update(strategy_features=torch.tensor(strategy_features(row['public_request']), dtype=torch.float32),
                           strategy_feature_names=list(NAMES), strategy_feature_version=VERSION)
                trace.append(row)
        return chosen

    def _row(self, env, seed, seat, old_opponent, replica, trace_count):
        row = super()._row(env, seed, seat, old_opponent, replica, trace_count)
        ledger = settlement_ledger(env.mjai_log, env.scores())
        row['point_components'] = ledger['components'][seat]
        row['settlement_ledger_passed'] = True
        if sum(row['point_components'].values()) != row['score'] - 25000:
            raise ValueError('Point breakdown does not equal net reward')
        self.latest_audited_outcome = row
        return row

    def _intervene(self, seed, seat, old_opponent, replica):
        result, rows = super()._intervene(seed, seat, old_opponent, replica)
        if not result['reused_control_outcome']:
            audited = self.latest_audited_outcome
            if (audited['seed'], audited['seat'], audited['replica']) != (seed, seat, replica):
                raise ValueError('Branch settlement audit identity mismatch')
            result['point_components'] = dict(audited['point_components'])
            result['settlement_ledger_passed'] = audited['settlement_ledger_passed']
        if sum(result['point_components'].values()) != result['score'] - 25000:
            raise ValueError('Branch/control ledger metadata misaligned')
        return result, rows
