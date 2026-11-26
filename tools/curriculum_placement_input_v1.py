"""One public-placement objective for all late decision types, same model.

This explicitly changes late ordinary instructions. It is a new research
contract, not ordinary-input parity with the parent on those windows.
"""
import copy

from tools.curriculum_endgame import in_endgame_scope
from tools.curriculum_rank_win_input_v1 import serving_request as win_projection, RankWinPolicyAgent

SCHEMA = 'all_late_terminal_top2_instruction_public_win_source_v1'
METHOD = 'all_late_same_model_complete_legal_RI_and_rank_win_route_v1'
INSTRUCTION = (
    'Choose a legal action in four-player Mahjong Soul East-only riichi mahjong. '
    'Prioritize the probability of finishing the complete match in the top two, '
    'then better terminal rank. If these placement prospects tie while currently '
    'third or fourth, consider net points; leading placement ties favor taking '
    'a legal win or retaining the ordinary continuation. Consider current public '
    'scores, dealer repeats, extensions and who pays a ron. A small win can be '
    'useful; the current hand is not assumed to end the match.'
)


def serving_request(request, normalizer=None):
    public = win_projection(request, normalizer=normalizer)
    if not in_endgame_scope(public['state']):
        return public
    q = public['questions']['action']
    # Preserve the pinned feature definitions/call note, removing the earlier
    # hand-win-first priority. Normalization is idempotent and label-independent.
    previous = q['instructions']
    marker = 'Consider legal yaku'
    suffix = previous[previous.index(marker):] if marker in previous else ''
    q['instructions'] = INSTRUCTION+(' '+suffix if suffix else '')
    return public


def training_record(row, normalizer=None):
    result = serving_request(row, normalizer=normalizer)
    if '_meta' in row: result['_meta'] = copy.deepcopy(row['_meta'])
    for name, q in row['questions'].items():
        for field in ('label','target','src'):
            if field in q: result['questions'][name][field] = copy.deepcopy(q[field])
    return result


class PlacementPolicyClient:
    def __init__(self, model): self.model = model
    def system_one(self, **request):
        return self.model.system_one(**serving_request(request))


def agent(model, validator=None):
    # Same execution and pointer parameters as the explicit win route. Only
    # the shared client projection changes instructions on late non-win input.
    return RankWinPolicyAgent(PlacementPolicyClient(model), validator=validator)
