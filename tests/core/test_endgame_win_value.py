import unittest
from jevmahjong.endgame.context import model_request
from jevmahjong.endgame.win_value import quote_win, allocate_groups, tiles


def position(red=False, payer=2):
    return {"state": {"seat": 1, "dealer": 3, "round_wind": 0, "hand_number": 4,
        "honba": 0, "sticks": 0, "scores": [32000, 22000, 25000, 21000],
        "hand": "2m 3m 4p " + ("0p" if red else "5p") + " 6p 6s 7s 8s 2p 2p",
        "dora_indicators": "9m", "riichi": [], "rivers": ["", "", "", ""],
        "melds": [[], [{"type": 1, "tiles": "5z 5z 5z"}], [], []],
        "last_discard": {"seat": payer, "tile": "4m"}},
        "questions": {"action": {"type": "choice", "criteria": {
            "a0": {"type": "hora"}, "a1": {"type": "none"}}}}}


class PublicWinValueTests(unittest.TestCase):
    def quote(self, record):
        return quote_win(model_request(record))

    def test_one_thousand_win_does_not_close_three_thousand_gap(self):
        result = self.quote(position())
        scenario = result["minimum_baseline_scenario"]
        self.assertEqual((scenario["han"], scenario["fu"]), (1, 30))
        self.assertEqual(scenario["deltas"], [0, 1000, -1000, 0])
        self.assertEqual(scenario["rank_after"], 3)

    def test_two_thousand_direct_hit_differs_from_another_payer(self):
        direct = self.quote(position(red=True))["minimum_baseline_scenario"]
        other = self.quote(position(red=True, payer=0))["minimum_baseline_scenario"]
        self.assertEqual(direct["deltas"], [0, 2000, -2000, 0])
        self.assertEqual((direct["rank_after"], other["rank_after"]), (2, 3))

    def test_sticks_and_honba_are_conserved_and_paid_once(self):
        record = position()
        record["state"].update(honba=2, sticks=1, scores=[31000, 22000, 25000, 21000])
        delta = self.quote(record)["minimum_baseline_scenario"]["deltas"]
        self.assertEqual(delta, [0, 2600, -1600, 0])
        self.assertEqual(sum(delta), 1000)

    def test_tied_result_uses_original_seat_priority(self):
        first = position(); first['state']['scores'] = [32000, 22000, 24000, 22000]
        self.assertEqual(self.quote(first)['minimum_baseline_scenario']['rank_after'], 2)
        second = position(payer=1)
        second['state'].update(seat=2, scores=[32000, 24000, 22000, 22000],
                               melds=[[], [], [{'type': 1, 'tiles': '5z 5z 5z'}], []])
        self.assertEqual(self.quote(second)['minimum_baseline_scenario']['rank_after'], 3)

    def test_dealer_tsumo_uses_three_equal_payments(self):
        record = position()
        record['state'].update(seat=3, scores=[32000, 25000, 21000, 22000],
                               melds=[[], [], [], [{'type': 1, 'tiles': '5z 5z 5z'}]])
        record['state']['hand'] += ' 4m'
        record['questions']['action']['criteria']['a0']['pai'] = '4m'
        self.assertEqual(self.quote(record)['minimum_baseline_scenario']['deltas'], [-500, -500, -500, 1500])

    def test_riichi_quote_excludes_hidden_ura(self):
        record = position()
        record['state'].update(hand='2m 3m 4m 4p 5p 6p 6s 7s 8s 2p 2p 5z 5z',
                               melds=[[], [], [], []], riichi=[1],
                               last_discard={'seat': 2, 'tile': '5z'})
        quote = self.quote(record)
        self.assertIn('ura_dora', quote['excluded_unconfirmed_bonuses'])
        self.assertEqual(quote['minimum_baseline_scenario']['deltas'], [0, 2600, -2600, 0])

    def test_tsumo_charges_all_three_with_dealer_payment(self):
        record = position()
        record["state"]["hand"] += " 4m"
        record["questions"]["action"]["criteria"]["a0"]["pai"] = "4m"
        result = self.quote(record)
        self.assertEqual(result["win_method"], "tsumo")
        self.assertIsNone(result["ron_payer"])
        self.assertEqual(result["minimum_baseline_scenario"]["deltas"], [-300, 1100, -300, -500])

    def test_unknown_drawn_tile_is_a_scenario_range_not_an_invented_tile(self):
        record = position()
        record["state"]["hand"] += " 4m"
        result = self.quote(record)
        self.assertFalse(result["winning_tile_known"])
        self.assertGreaterEqual(result["scenario_count"], 1)

    def test_all_three_kan_meld_types_preserve_structure(self):
        for kind in (2, 3, 4):
            record = position()
            record["state"]["melds"][1] = [{"type": kind, "tiles": "5z 5z 5z 5z"}]
            self.assertEqual(self.quote(record)["status"], "public_single_winner_baseline_scenarios")

    def test_future_labels_cannot_change_public_quote(self):
        record = position()
        first = self.quote(record)
        record["_meta"] = {"actual_future_ura": ["1p"], "future_scores": [0, 100000, 0, 0]}
        record["questions"]["action"]["target"] = {"a0": 0., "a1": 1.}
        self.assertEqual(self.quote(record), first)

    def test_red_and_normal_tiles_never_share_a_physical_id(self):
        ids = allocate_groups([tiles("0555m")])[0]
        self.assertEqual(sorted(ids), [16, 17, 18, 19])
        with self.assertRaises(ValueError):
            allocate_groups([tiles("005m")])

    def test_v3_removes_unknown_future_screen_fields(self):
        from jevmahjong.endgame.win_value import settlement_request
        record = position()
        record["state"]["screen_observation"] = {"ura_dora_indicators": ["1p"],
            "future_scores": [0, 100000, 0, 0], "reaction": {"seat": 2, "tile": "4m", "future_yaku": "secret"}}
        payload = settlement_request(record)
        screen = payload["state"]["screen_observation"]
        self.assertNotIn("ura_dora_indicators", screen)
        self.assertNotIn("future_scores", screen)
        self.assertNotIn("future_yaku", screen["reaction"])
        self.assertEqual(payload["state"]["endgame_context"]["current_win_settlement"]["minimum_baseline_scenario"]["rank_after"], 3)

    def test_physical_public_river_preserves_native_ron_payer(self):
        import base64, json
        from types import SimpleNamespace
        from riichienv import ActionType
        from jevmahjong.endgame.native_public import add_native_context
        from jevmahjong.endgame.win_value import settlement_request
        request = position()
        request["state"]["last_discard"] = 12
        obs = SimpleNamespace(to_dict=lambda: {"discards": [[0], [1], [8, 12], [11]]},
            serialize_to_base64=lambda: base64.b64encode(json.dumps({"last_discard": 12}).encode()))
        actions = [SimpleNamespace(action_type=ActionType.RON, tile=12)]
        payload = settlement_request(add_native_context(request, obs, actions))
        self.assertEqual(payload["state"]["endgame_context"]["ron_payer"], 2)
        self.assertEqual(payload["state"]["endgame_context"]["public_winning_tile"], "4m")

    def test_known_tsumo_tile_is_preserved_without_a_future_event(self):
        import base64, json
        from types import SimpleNamespace
        from riichienv import ActionType
        from jevmahjong.endgame.native_public import add_native_context
        from jevmahjong.endgame.win_value import settlement_request
        request = position(); request["state"]["hand"] += " 4m"
        obs = SimpleNamespace(to_dict=lambda: {"discards": [[], [], [], []]},
            serialize_to_base64=lambda: base64.b64encode(json.dumps({"last_discard": 40}).encode()))
        payload = settlement_request(add_native_context(request, obs, [SimpleNamespace(action_type=ActionType.TSUMO, tile=12)]))
        quote = payload["state"]["endgame_context"]["current_win_settlement"]
        self.assertTrue(quote["winning_tile_known"])
        self.assertEqual(quote["win_method"], "tsumo")

    def test_pass_is_an_option_only_when_public_baseline_still_trails(self):
        from jevmahjong.endgame.win_value import settlement_request
        from jevmahjong.endgame.public_policy import action_keys
        small = settlement_request(position())
        enough = settlement_request(position(red=True))
        self.assertEqual(action_keys(small), ['a0', 'a1'])
        self.assertEqual(action_keys(enough), ['a0'])
        self.assertEqual(action_keys(settlement_request(position(red=True, payer=0))), ['a0', 'a1'])

    def test_dealer_or_unresolved_scoring_keeps_legal_win_available(self):
        from jevmahjong.endgame.win_value import settlement_request
        from jevmahjong.endgame.public_policy import action_keys
        dealer = position(); dealer['state']['dealer'] = 1
        self.assertEqual(action_keys(settlement_request(dealer)), ['a0'])
        unknown = settlement_request(position())
        unknown['state']['endgame_context']['current_win_settlement'] = {'status': 'unresolved'}
        self.assertEqual(action_keys(unknown), ['a0'])

    def test_public_guard_reconstructs_original_minimum_shanten_support(self):
        from jevmahjong.endgame.public_policy import original_policy_keys
        record = position()
        record['questions']['action']['criteria'] = {
            'bad': {'type': 'dahai', 'sh': 2, 'ukeire': 20, 'safe': [], 'red': False},
            'good': {'type': 'dahai', 'sh': 1, 'ukeire': 10, 'safe': [], 'red': False},
            'kan': {'type': 'ankan', 'consumed': ['5z'] * 4}}
        self.assertEqual(original_policy_keys(record), ['good', 'kan'])


if __name__ == "__main__":
    unittest.main()
