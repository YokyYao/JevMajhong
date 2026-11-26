"""Final comparison's shared native financial checks and unchanged v8 adapter."""
from tools.curriculum_confirmation_stats_v1 import matrix


class FrozenReferenceEventAdapter:
    """Keep original PolicyAgent.act(obs) semantics in the explicit-event loop."""
    def __init__(self,original):self.original=original
    def act(self,obs,*,current_public_event=None):return self.original.act(obs)


def audit_matches(records,seed_start,settlement_ledger,hand_dealin_count):
    matrix(records,seed_start)
    for row in records:
        seat=row['seat'];log=row['mjai_log'];ledger=settlement_ledger(log,row['final_scores'])
        if (ledger['hands']!=row['hands']
            or row['wins']!=sum(e['type']=='hora' and e.get('actor')==seat for e in log)
            or row['dealin_hands']!=hand_dealin_count(log,seat)):
            raise ValueError('Actual native ledger/hand/win/dealin counters differ')
        if set(row['choices'])!={'chi','pon','ankan','daiminkan','kakan','reach'}:
            raise ValueError('All original call/kan/riichi event counts required')
        for operation,n in row['choices'].items():
            if type(n) is not int or n<0 or n!=sum(e['type']==operation and e.get('actor')==seat for e in log):
                raise ValueError('Actual native call/kan/riichi event count differs')
    return dict(complete_matches=len(records),every_native_financial_ledger_reconstructed=True,
        every_rank_net_hand_win_dealin_operation_counter_checked=True,
        matrix_unit='32_logical_seeds_four_seats_two_profiles',
        model_policy_identity_and_source_provenance_not_verified_by_this_function=True)
