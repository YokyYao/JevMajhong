import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from desktop.jev_sidecar.engine import Adviser, Board, candidates, request
from desktop.jev_sidecar.tiles import parse
from desktop.jev_sidecar.endgame_policy import EndgamePolicyClient, canonical_request
from desktop.jev_sidecar.unified_policy import UnifiedPolicyClient


class Model:
    def __init__(self):
        self.requests = []

    def system_one(self, **payload):
        self.requests.append(payload)
        criteria = payload["questions"]["action"]["criteria"]
        chosen = next((k for k, v in criteria.items() if v["type"] == "none"), next(iter(criteria)))
        return {"answers": {"action": {"probabilities": {k: float(k == chosen) for k in criteria}}}}


def validation_stub(probabilities, keys):
    return [probabilities[k] for k in keys], {}


def winning_board(hand_number=4, confirmed=True):
    return Board(parse("123m123p123s1112z"), rivers=[[], [], [], []],
        scores=[25000, 20000, 30000, 25000], dealer=2, hand_number=hand_number,
        metadata_confirmed=confirmed, reaction={"tile": "2z", "seat": 1, "offered": ["hora"]})


def small_win_board():
    return Board(parse('23m456p678s22p'), rivers=[[], [], [], []],
        indicators=['9m'], melds=[[{'type': 1, 'tiles': '5z 5z 5z'}], [], [], []],
        scores=[22000, 25000, 32000, 21000], dealer=2, hand_number=4,
        metadata_confirmed=True, reaction={'tile': '4m', 'seat': 1, 'offered': ['hora']})


class EndgameDesktopTests(unittest.TestCase):
    def test_starting_east_mapping_resolves_score_tie_and_safety_seats(self):
        board = Board(parse("12389m278p589s457z"), rivers=[[], [], [], []],
            scores=[25000, 20000, 30000, 25000], dealer=2, hand_number=4,
            metadata_confirmed=True, riichi=[1])
        source = request(board, candidates(board))
        source["questions"]["action"]["criteria"]["a0"]["safe"] = [1]
        before = copy.deepcopy(source)
        canonical, mapping = canonical_request(source)
        self.assertEqual(source, before)
        self.assertEqual(mapping, {3: 0, 0: 1, 1: 2, 2: 3})
        self.assertEqual(canonical["state"]["seat"], 1)
        self.assertEqual(canonical["state"]["scores"], [25000, 25000, 20000, 30000])
        self.assertEqual(canonical["state"]["riichi"], [2])
        self.assertEqual(canonical["questions"]["action"]["criteria"]["a0"]["safe"], [2])
        model = Model()
        answer, trace = EndgamePolicyClient(model).system_one_with_trace(**source)
        context = model.requests[0]["state"]["endgame_context"]
        self.assertEqual(context["placement"], 3)
        self.assertEqual(context["target_seat"], 0)
        self.assertEqual(context["gap_to_target"], 0)
        self.assertEqual(trace["relative_to_starting_east"], mapping)

    def test_confirmed_late_win_pass_uses_one_model_inference(self):
        model = Model()
        adviser = Adviser()
        adviser.model = EndgamePolicyClient(model)
        adviser.policy_name = "Jev Endgame v1"
        api = SimpleNamespace(validate_distribution=validation_stub)
        with patch.dict(sys.modules, {"jevany": SimpleNamespace(__path__=[]), "jevany.api": api}):
            result = adviser.recommend(winning_board(), use_model=True)
        self.assertEqual(result["top3"][0]["action"]["type"], "none")
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(result["model_input"]["route"], "endgame")
        self.assertEqual(result["model_input"]["model_request"]["state"]["seat"], 1)

    def test_earlier_or_unconfirmed_win_keeps_existing_take_win_behavior(self):
        for board in (winning_board(1), winning_board(4, False)):
            model = Model()
            adviser = Adviser()
            adviser.model = EndgamePolicyClient(model)
            result = adviser.recommend(board, use_model=True)
            self.assertEqual(result["top3"][0]["action"]["type"], "hora")
            self.assertEqual(model.requests, [])

    def test_released_play_v2_win_behavior_is_unchanged(self):
        model = Model()
        adviser = Adviser()
        adviser.model = UnifiedPolicyClient(model)
        result = adviser.recommend(winning_board(), use_model=True)
        self.assertEqual(result["top3"][0]["action"]["type"], "hora")
        self.assertEqual(model.requests, [])



    def test_legacy_pair_form_and_screen_ron_payer_features_match(self):
        from jevmahjong.endgame.context import model_request
        model = Model()
        adviser = Adviser()
        adviser.model = EndgamePolicyClient(model)
        api = SimpleNamespace(validate_distribution=validation_stub)
        with patch.dict(sys.modules, {"jevany": SimpleNamespace(__path__=[]), "jevany.api": api}):
            result = adviser.recommend(winning_board(), use_model=True)
        actual = result["model_input"]["model_request"]
        native = copy.deepcopy(actual)
        native["state"].pop("endgame_context")
        native["state"]["last_discard"] = [2, 112]
        native["state"]["screen_observation"].pop("reaction", None)
        normalized = model_request(native)
        self.assertEqual(actual["state"]["last_discard"], normalized["state"]["last_discard"])
        for key in ("current_discarder", "public_discard_tile", "win_method", "ron_payer"):
            self.assertEqual(actual["state"]["endgame_context"][key],
                             normalized["state"]["endgame_context"][key])
        self.assertEqual(actual["state"]["endgame_context"]["ron_payer"], 2)
        self.assertEqual(actual["state"]["endgame_context"]["win_method"], "ron")

    def test_v3_scores_native_physical_discard_and_screen_reaction_identically(self):
        import base64, json
        try:
            from riichienv import ActionType
        except ModuleNotFoundError as exc:
            if exc.name != "riichienv":
                raise
            self.skipTest("Native comparison runs in the training checkout")
        from jevmahjong.endgame.native_public import add_native_context
        from jevmahjong.endgame.win_value import settlement_request
        model = Model(); adviser = Adviser()
        adviser.model = EndgamePolicyClient(model, input_version="endgame-public-v3")
        api = SimpleNamespace(validate_distribution=validation_stub)
        with patch.dict(sys.modules, {"jevany": SimpleNamespace(__path__=[]), "jevany.api": api}):
            result = adviser.recommend(small_win_board(), use_model=True)
        actual = result['model_input']['model_request']
        native = copy.deepcopy(actual)
        native['state']['last_discard'] = 12
        native['state']['screen_observation'].pop('reaction', None)
        native['state']['screen_observation'].pop('winning_tile', None)
        obs = SimpleNamespace(to_dict=lambda: {'discards': [[], [], [12], []]},
            serialize_to_base64=lambda: base64.b64encode(json.dumps({'last_discard': 12}).encode()))
        normalized = settlement_request(add_native_context(native, obs, [SimpleNamespace(action_type=ActionType.RON, tile=12)]))
        for key in ('ron_payer', 'public_winning_tile', 'current_win_settlement'):
            self.assertEqual(actual['state']['endgame_context'][key], normalized['state']['endgame_context'][key])
        self.assertEqual(len(model.requests), 1)

    def test_v3_does_not_treat_a_display_fallback_as_a_known_drawn_tile(self):
        for drawn in (None, '4m'):
            board = small_win_board()
            board.hand.append('4m'); board.drawn = drawn
            board.reaction = None; board.self_win_available = True
            model = Model(); adviser = Adviser()
            adviser.model = EndgamePolicyClient(model, input_version='endgame-public-v3')
            api = SimpleNamespace(validate_distribution=validation_stub)
            with patch.dict(sys.modules, {'jevany': SimpleNamespace(__path__=[]), 'jevany.api': api}):
                result = adviser.recommend(board, use_model=True)
            context = result['model_input']['model_request']['state']['endgame_context']
            self.assertEqual(context['current_win_settlement']['winning_tile_known'], drawn is not None)
            self.assertEqual(context['public_winning_tile'], drawn)
            self.assertEqual(len(model.requests), 1)

    def test_v3_takes_a_sufficient_or_leading_win_without_model_inference(self):
        leading = small_win_board(); leading.scores = [32000, 25000, 22000, 21000]
        dealer = small_win_board(); dealer.dealer = 0
        for board in (winning_board(), leading, dealer):
            model = Model(); adviser = Adviser()
            adviser.model = EndgamePolicyClient(model, input_version='endgame-public-v3')
            result = adviser.recommend(board, use_model=True)
            self.assertEqual(result['top3'][0]['action']['type'], 'hora')
            self.assertEqual(model.requests, [])



    def test_tsumo_is_offered_only_with_a_real_button_and_complete_shape(self):
        from desktop.jev_sidecar.engine import candidates
        board = Board(parse("123m123p123s11122z"), rivers=[[], [], [], []], drawn="2z")
        self.assertFalse(any(v.get("action", {}).get("type") == "hora" for v in candidates(board)))
        board.self_win_available = True
        self.assertTrue(any(v.get("action", {}).get("type") == "hora" for v in candidates(board)))
        board.hand = parse("12389m278p589s457z")
        self.assertFalse(any(v.get("action", {}).get("type") == "hora" for v in candidates(board)))

    def test_late_tsumo_can_continue_and_is_explained_as_a_declined_win(self):
        board = Board(parse("123m123p123s11122z"), rivers=[[], [], [], []], drawn="2z",
            self_win_available=True, scores=[25000, 20000, 30000, 25000],
            dealer=2, hand_number=4, metadata_confirmed=True)
        model = Model()
        adviser = Adviser()
        adviser.model = EndgamePolicyClient(model)
        api = SimpleNamespace(validate_distribution=validation_stub)
        with patch.dict(sys.modules, {"jevany": SimpleNamespace(__path__=[]), "jevany.api": api}):
            result = adviser.recommend(board, use_model=True)
        self.assertTrue(result["own_win_available"])
        self.assertTrue(result["endgame_decision"])
        self.assertNotEqual(result["top3"][0].get("action", {}).get("type"), "hora")
        self.assertIn("放弃本次自摸", result["decision"])
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(model.requests[0]["state"]["endgame_context"]["win_method"], "tsumo")
        self.assertIsNone(model.requests[0]["state"]["endgame_context"]["ron_payer"])

    def test_current_v2_takes_tsumo_without_a_new_model_call_and_has_a_click_plan(self):
        from desktop.jev_sidecar.autoplay import plan_click
        board = Board(parse("123m123p123s11122z"), rivers=[[], [], [], []], drawn="2z",
            self_win_available=True, metadata_confirmed=True, hand_number=4)
        model = Model()
        adviser = Adviser()
        adviser.model = UnifiedPolicyClient(model)
        result = adviser.recommend(board, use_model=True)
        self.assertEqual(result["top3"][0]["action"]["type"], "hora")
        self.assertEqual(model.requests, [])
        plan = plan_click(result, [], {"offered": ["hora"], "self_win_available": True,
            "reaction": None}, [0, 0, 1, 1], (1280, 720))
        self.assertIsNotNone(plan)
        self.assertEqual(plan.label, "自摸和牌")


if __name__ == "__main__":
    unittest.main()

