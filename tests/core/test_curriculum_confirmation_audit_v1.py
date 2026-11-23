import copy
import unittest

from tools.audit_curriculum_confirmation_v1 import require_complete_outcomes,verify_seed_prefixes
from tools.curriculum_confirmation_runtime_v1 import STATUS,REFERENCE
from tools.curriculum_confirmation_contract_v1 import CRITERIA


class ConfirmationAuditTests(unittest.TestCase):
    def test_a_claimed_complete_count_cannot_replace_the_actual_two_matrices(self):
        reg=dict(status=STATUS,criteria=copy.deepcopy(CRITERIA),game_mode='4p-red-east',rule='GameRule.default_mjsoul',
            seed_namespace='spaced-v1',role_order=['reference','candidate'],
            candidate_model=dict(identity_sha256='candidate'),reference_model=dict(identity_sha256=REFERENCE),
            candidate_policy_identity='candidate-policy',reference_policy_identity='reference-policy',
            optimizer_updates_during_confirmation=0,reserved_grid_already_opened=False,automatic_release=False,
            allow_adaptive_extra_games=False,raw_matches_private=True,requires_independent_postrun_audit=True)
        opened=dict(registered_sha256='seal',candidate_identity='candidate',reference_identity=REFERENCE,
            seed_start=2026300000,seed_groups=32,scheduled_matches_per_policy=256,no_optimizer_updates=True)
        result=dict(status='actual_fixed512_confirmation_complete_pending_independent_audit',registered_sha256='seal',
            actual_complete_games=512,optimizer_updates=0,sole_candidate_unchanged=True,
            independent_postrun_audit_still_required=True,
            models={role:reg[role+'_model'] for role in ('candidate','reference')},
            policy_identities={role:reg[role+'_policy_identity'] for role in ('candidate','reference')})
        with self.assertRaises(ValueError):require_complete_outcomes(reg,opened,result,dict(reference=[],candidate=[]),'seal')
        partial=copy.deepcopy(result);partial['actual_complete_games']=511
        with self.assertRaises(ValueError):require_complete_outcomes(reg,opened,partial,dict(reference=[],candidate=[]),'seal')
        changed=copy.deepcopy(opened);changed['candidate_identity']='different'
        with self.assertRaises(ValueError):require_complete_outcomes(reg,changed,result,dict(reference=[],candidate=[]),'seal')

    def test_every_match_checks_the_native_seed_prefix_and_reuses_only_same_seed_reset(self):
        def row(seed):return dict(seed=seed,mjai_log=[dict(type='synthetic_start',seed=seed),dict(type='synthetic_next')])
        records=dict(reference=[row(910000000),row(910000001)],candidate=[row(910000000),row(910000001)])
        calls=[]
        def prefix(seed):calls.append(seed);return [dict(type='synthetic_start',seed=seed)]
        checked=verify_seed_prefixes(records,prefix)
        self.assertEqual(calls,[910000000,910000001])
        self.assertEqual(checked['actual_match_prefixes_checked'],4)
        bad=copy.deepcopy(records);bad['candidate'][-1]['mjai_log'][0]['seed']=0
        with self.assertRaises(ValueError):verify_seed_prefixes(bad,prefix)
        with self.assertRaises(ValueError):verify_seed_prefixes(records,lambda seed:[])
