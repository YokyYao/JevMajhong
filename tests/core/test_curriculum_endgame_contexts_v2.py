import unittest
from tools.curriculum_endgame_contexts import starting_scores as old_starting_scores
from tools.curriculum_endgame_contexts_v2 import accounting_history,starting_scores


class LiveAccountingTests(unittest.TestCase):
    def test_post_chi_pon_kan_prefixes_conserve_the_same_deposits(self):
        for kind in ('chi','pon','daiminkan','kakan','ankan'):
            history=[dict(type='start_kyoku'),dict(type='reach',actor=1),dict(type='reach_accepted',actor=1),dict(type=kind,actor=2)]
            self.assertEqual(starting_scores(history,[25000,24000,25000,25000],1),([25000]*4,0))

    def test_original_win_scopes_keep_exact_original_accounting(self):
        for kind in ('dahai','tsumo'):
            last=dict(type=kind,actor=2,pai='1m',tsumogiri=True)
            history=[dict(type='start_kyoku'),dict(type='reach_accepted',actor=1),last]
            self.assertEqual(starting_scores(history,[25000,24000,25000,25000],1),old_starting_scores(history,[25000,24000,25000,25000],1))

    def test_only_current_hand_deposits_are_reversed(self):
        history=[dict(type='start_kyoku'),dict(type='reach_accepted',actor=0),dict(type='hora'),
                 dict(type='start_kyoku'),dict(type='pon',actor=2)]
        self.assertEqual(starting_scores(history,[25000]*4,0),([25000]*4,0))

    def test_settled_or_unknown_current_prefix_is_rejected(self):
        for kind in ('hora','ryukyoku','end_kyoku','end_game','unknown'):
            with self.assertRaises(ValueError):accounting_history([dict(type='start_kyoku'),dict(type=kind,actor=1)])

    def test_invalid_actor_duplicate_deposit_and_nonconservation_rejected(self):
        for history,scores,bank in [([dict(type='start_kyoku'),dict(type='pon',actor=True)],[25000]*4,0),
            ([dict(type='start_kyoku'),dict(type='reach_accepted',actor=1),dict(type='reach_accepted',actor=1)],[25000]*4,2),
            ([dict(type='start_kyoku'),dict(type='chi',actor=1)],[25000]*4,1)]:
            with self.assertRaises(ValueError):starting_scores(history,scores,bank)

    def test_accounting_projection_never_copies_private_hands_or_draw_identities(self):
        history=[dict(type='start_kyoku',tehais=[['1m']]*4),dict(type='tsumo',actor=2,pai='2m')]
        self.assertEqual(accounting_history(history),[dict(type='start_kyoku'),dict(type='tsumo',actor=2)])


if __name__=='__main__':unittest.main()
