"""Discard features and the trained Jev interface for screenshot-derived state."""
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import re
import os
from itertools import combinations
from pathlib import Path

from mahjong.shanten import Shanten
from .tiles import dora, index, mjai, name, normal, parse

CHECKPOINT_SHA = 'b1248fbc9c9bd1a768a9eaf5fdb5241c82c96ac50eaabf8e0ca8fed885bec1ea'
INSTRUCTION = ("Choose the best legal action in four-player Mahjong Soul riichi mahjong. "
    "Maximize final placement: pursue first while avoiding costly deal-ins and last place. "
    "sh=shanten (lower is closer to tenpai); ukeire=visible remaining improving tiles; "
    "safe=opponent seats for which this tile is genbutsu. Probabilities are action preferences.")


def parse_melds(text):
    result = []
    for group in re.split(r'[,;|，\s]+', text.strip()):
        if not group:
            continue
        prefix = group[:2] if group[:2] in ('a:', 'k:') else ''
        tiles = parse(group[2:] if prefix else group)
        types = sorted(index(t) for t in tiles)
        if len(tiles) == 4 and len(set(types)) == 1:
            kind = 3 if prefix == 'a:' else 4 if prefix == 'k:' else 2
        elif len(tiles) == 3 and len(set(types)) == 1 and not prefix:
            kind = 1
        elif len(tiles) == 3 and types[0] < 27 and types[0] // 9 == types[-1] // 9 and types == list(range(types[0], types[0] + 3)) and not prefix:
            kind = 0
        else:
            raise ValueError('副露格式：123m,555p；暗杠a:1111z，加杠k:5555p')
        result.append({'type': kind, 'tiles': ' '.join(tiles)})
    if len(result) > 4:
        raise ValueError('每家最多四组副露')
    return result


@lru_cache(maxsize=100000)
def shanten(types):
    counts = [0] * 34
    for value in types:
        counts[value] += 1
    closed = len(types) in (13, 14)
    return Shanten.calculate_shanten(counts, use_chiitoitsu=closed, use_kokushi=closed)


@dataclass
class Board:
    hand: list[str]
    rivers: list = field(default_factory=lambda: [None] * 4)
    indicators: list[str] = field(default_factory=list)
    melds: list = field(default_factory=lambda: [[], [], [], []])
    riichi: list[int] = field(default_factory=list)
    scores: list[int] = field(default_factory=lambda: [25000] * 4)
    seat: int = 0
    dealer: int = 0
    round_wind: int = 0
    honba: int = 0
    sticks: int = 0
    drawn: str | None = None
    metadata_confirmed: bool = False
    hand_number: int | None = None
    reaction: dict | None = None
    forbidden_discards: list[str] = field(default_factory=list)
    riichi_available: bool = False
    riichi_selecting: bool = False
    self_kan_available: bool = False

    def visible(self):
        if len(self.rivers) != 4 or len(self.melds) != 4 or len(self.scores) != 4:
            raise ValueError('需要四家的牌河、副露和分数')
        values = list(self.hand) + list(self.indicators)
        for river in self.rivers:
            values.extend(river or [])
        for groups in self.melds:
            for group in groups:
                values.extend(parse(group['tiles']))
        counts = Counter(index(t) for t in values)
        if any(n > 4 for n in counts.values()):
            raise ValueError('手牌与公开牌面出现同牌超过4张，请修正牌河/副露/宝牌')
        return counts

    def notes(self):
        result = []
        missing = sum(r is None for r in self.rivers)
        if missing:
            result.append(f'{missing}家牌河未确认：进张为上界，防守信息不完整')
        if not self.metadata_confirmed:
            result.append('分数、立直与副露尚未确认')
        result.append('截图不提供完整时序；建议依据当前可见牌面')
        return result


def hand_strategy(board, choice, direction_only=False):
    """Explain a plausible direction from the selected action, not model reasoning."""
    action = choice.get('action', {}).get('type', 'dahai')
    if action == 'hora':
        return '当前可和牌，优先完成本局。'
    hand = list(board.hand)
    if action in ('dahai', 'reach') and choice.get('tile') in hand:
        hand.remove(choice['tile'])
    groups = board.melds[board.seat]
    all_tiles = hand + [t for g in groups for t in parse(g['tiles'])]
    counts = Counter(index(t) for t in all_tiles)
    closed = all(g['type'] == 3 for g in groups)
    pairs = sum(n >= 2 for n in counts.values())
    triples = sum(n >= 3 for n in counts.values())
    suits = Counter(t[1] for t in all_tiles if t[1] != 'z')
    honors = sum(t.endswith('z') for t in all_tiles)
    dragons = [name(str(i - 26) + 'z') for i in (31, 32, 33) if counts[i] >= 2]
    if set(board.riichi) - {board.seat} and choice.get('sh', 0) and choice['sh'] >= 1:
        plan = '防守优先，先减少放铳风险，再寻找进张'
    elif board.seat in board.riichi:
        plan = '已立直，保持听牌，等待和牌'
    elif closed and pairs >= 5:
        plan = '对子较多，可朝七对子推进，同时保留普通面子手'
    elif suits and max(suits.values()) >= 10 and sum(suits.values()) - max(suits.values()) <= 2:
        suit_name = {'m': '万', 'p': '筒', 's': '条'}[suits.most_common(1)[0][0]]
        plan = f'牌集中在{suit_name}，可朝' + ('混一色' if honors else '清一色') + '推进'
    elif dragons:
        plan = '保留' + '、'.join(dragons) + '对子或刻子，朝役牌手推进'
    elif triples >= 2 and pairs >= 4 and not any(g['type'] == 0 for g in groups):
        plan = '对子、刻子较多，可朝对对和推进'
    elif all(i < 27 and i % 9 not in (0, 8) for i in counts):
        plan = '以中张组成面子，朝断幺九推进' + ('，保留门清立直机会' if closed else '')
    else:
        plan = '优先补齐面子与雀头，兼顾立直与有役副露路线' if closed else '优先补齐面子与雀头，保留有役副露路线'
    if direction_only:
        return plan
    if action == 'none':
        return '跳过这次鸣牌；' + plan + '。'
    if action in ('ankan', 'kakan'):
        return choice['name'] + '；杠后补牌，再重新判断弃牌与和牌路线。'
    if action in ('chi', 'pon', 'daiminkan'):
        return '鸣牌后重新整理弃牌；需保留役牌等有役路线。'
    distance = choice.get('sh')
    if distance == 0:
        waits = [name(str(i % 9 + 1) + 'mpsz'[i // 9]) for i in choice.get('improving', [])]
        return plan + '；听牌等 ' + '、'.join(waits) + '。'
    return plan + (f'；弃牌后{distance}向听。' if distance is not None else '。')


def decision_description(board, choice):
    """Observable consequences of this choice, separate from the hand's plan."""
    action = choice.get('action', {}).get('type', 'dahai')
    if action == 'none':
        return '跳过这次吃碰，保留现有手牌。'
    if action == 'hora':
        return '和牌，完成本局。'
    if action in ('ankan', 'kakan'):
        return choice['name'] + '；杠后补牌，再重新判断弃牌。'
    if action in ('chi', 'pon', 'daiminkan'):
        return choice['name'] + '；鸣牌后继续选择弃牌，需保留有役路线。'
    detail = ('立直后打 ' if action == 'reach' else '打 ') + name(choice['tile'])
    if choice.get('sh') == 0:
        waits = [name(str(i % 9 + 1) + 'mpsz'[i // 9]) for i in choice.get('improving', [])]
        detail += '；听牌等 ' + '、'.join(waits)
    elif choice.get('sh') is not None:
        detail += f'；弃牌后{choice["sh"]}向听，进张{choice["ukeire"]}枚'
    if set(choice.get('safe', [])) & (set(board.riichi) - {board.seat}):
        detail += '；是立直家的现物'
    return detail + '。'


def candidates(board):
    n = len(board.hand)
    if n not in (2, 5, 8, 11, 14):
        raise ValueError(f'当前手牌{n}张，等待轮到自己出牌；副露后应为11/8/5/2张')
    parse(' '.join(board.hand))
    expected_melds = (14 - n) // 3
    if len(board.melds[board.seat]) != expected_melds:
        raise ValueError('手牌张数与自己的副露组数不一致，请在牌桌信息中校正')
    visible = board.visible()
    if board.seat in board.riichi and (not board.drawn or board.drawn not in board.hand):
        raise ValueError('自己已立直：请确认最右侧摸入牌')
    values = []
    tiles = sorted(set(board.hand), key=lambda t: json.dumps({'pai': mjai(t), 'type': 'dahai'}, sort_keys=True))
    for tile in tiles:
        if normal(tile) in {normal(t) for t in board.forbidden_discards}:
            continue
        if board.seat in board.riichi and tile != board.drawn:
            continue
        remaining = list(board.hand)
        remaining.remove(tile)
        types = tuple(sorted(index(t) for t in remaining))
        distance = shanten(types)
        improving = [i for i in range(34) if visible[i] < 4 and shanten(tuple(sorted((*types, i)))) < distance]
        safe = [p for p, river in enumerate(board.rivers) if p != board.seat and river is not None
                and index(tile) in {index(t) for t in river}]
        values.append({'tile': tile, 'name': name(tile), 'sh': distance,
                       'ukeire': sum(4 - visible[t] for t in improving), 'improving': improving,
                       'safe': safe, 'red': tile[0] == '0', 'dora': sum(index(tile) == dora(t) for t in board.indicators)})
    if not values:
        raise ValueError('正在读取当前合法弃牌')
    threats = set(board.riichi) - {board.seat}
    fold = bool(threats) and min(v['sh'] for v in values) >= 1
    ref = min(values, key=lambda v: (len(threats - set(v['safe'])) if fold else 0, v['sh'], -v['ukeire'], v['red'], tiles.index(v['tile'])))
    eligible = [v for v in values if v['sh'] <= ref['sh'] and
                (not fold or len(threats - set(v['safe'])) <= len(threats - set(ref['safe'])))]
    for value in values:
        value['eligible'] = value in eligible
    if board.riichi_selecting:
        values = [v for v in values if v['sh'] == 0]
        if not values:
            raise ValueError('立直选牌中，正在重新读取听牌弃牌')
    elif board.riichi_available and board.seat not in board.riichi and not board.melds[board.seat]:
        values += [{**v, 'name': '立直 → 打 ' + v['name'],
                    'action': {'type': 'reach', 'discard': mjai(v['tile'])}}
                   for v in list(values) if v['sh'] == 0]
    if board.self_kan_available and not board.riichi_selecting:
        values += self_kan_candidates(board)
    return values


def self_kan_candidates(board):
    """Own-turn kans are considered only when the game offers the kan button."""
    if not board.self_kan_available:
        return []
    actions = []
    counts = Counter(index(t) for t in board.hand)
    for tile_type, count in sorted(counts.items()):
        if count != 4:
            continue
        consumed = [t for t in board.hand if index(t) == tile_type]
        if board.seat in board.riichi:
            # The visible button can represent several choices. Only retain a
            # drawn-tile ankan which preserves the complete set of waits.
            if not board.drawn or index(board.drawn) != tile_type:
                continue
            before = list(board.hand)
            before.remove(board.drawn)
            after = [t for t in board.hand if index(t) != tile_type]
            def waits(hand):
                types = [index(t) for t in hand]
                return {t for t in range(34) if types.count(t) < 4 and
                        shanten(tuple(sorted(types + [t]))) == -1}
            if not waits(before) or waits(before) != waits(after):
                continue
        actions.append({'type': 'ankan', 'pai': mjai(consumed[0]),
                        'consumed': sorted(mjai(t) for t in consumed)})
    if board.seat not in board.riichi:
        for group in board.melds[board.seat]:
            if group['type'] != 1:
                continue
            tiles = parse(group['tiles'])
            for tile in sorted(set(board.hand)):
                if index(tile) == index(tiles[0]):
                    actions.append({'type': 'kakan', 'pai': mjai(tile),
                                    'consumed': sorted(mjai(t) for t in tiles)})
    return [{'action': action, 'name': ('暗杠 ' if action['type'] == 'ankan' else '加杠 ') +
             name(next(t for t in board.hand if mjai(t) == action['pai'])),
             'tile': next(t for t in board.hand if mjai(t) == action['pai']),
             'sh': None, 'ukeire': None, 'safe': [], 'red': False, 'eligible': True}
            for action in actions]


def request(board, values):
    state = {'seat': board.seat, 'dealer': board.dealer, 'round_wind': board.round_wind,
             'honba': board.honba, 'sticks': board.sticks, 'scores': board.scores,
             'hand': ' '.join(sorted(board.hand, key=lambda t: (index(t), t))),
             'dora_indicators': ' '.join(board.indicators), 'riichi': board.riichi,
             'rivers': [None if r is None else ' '.join(r) for r in board.rivers], 'melds': board.melds,
             'discards_count': sum(len(r or []) for r in board.rivers),
             'hand_number': board.hand_number if board.hand_number is not None else board.dealer + 1,
             'tsumogiri_flags': None, 'riichi_sutehais': None, 'last_tedashis': None, 'last_discard': None,
             'screen_observation': {'temporal_information': 'unknown', 'metadata_confirmed': board.metadata_confirmed}}
    criteria = {f'a{i}': {**v.get('action', {'type': 'dahai', 'pai': mjai(v['tile'])}),
                         **{key: v[key] for key in ['sh', 'ukeire', 'safe', 'red']}} for i, v in enumerate(values)}
    return {'state': state, 'questions': {'action': {'type': 'choice', 'instructions': INSTRUCTION, 'criteria': criteria}}}


def reaction_candidates(board):
    """Only enumerate calls offered by the visible game buttons."""
    event = board.reaction
    if not event:
        return []
    parse(' '.join(board.hand))
    board.visible()
    tile, offered = event['tile'], set(event['offered'])
    values = [{'action': {'type': 'none'}, 'name': '跳过', 'tile': None, 'sh': None, 'ukeire': None, 'safe': [], 'eligible': True}]
    for kind, count in [('chi', 2), ('pon', 2), ('daiminkan', 3)]:
        if kind not in offered:
            continue
        seen = set()
        for consumed in combinations(board.hand, count):
            signature = tuple(sorted(consumed))
            if signature in seen:
                continue
            seen.add(signature)
            types = sorted(index(t) for t in (*consumed, tile))
            legal = (kind != 'chi' and len(set(types)) == 1) or (kind == 'chi' and event.get('seat') == 3
                and types[0] < 27 and types[0] // 9 == types[-1] // 9 and types == list(range(types[0], types[0] + 3)))
            if not legal:
                continue
            label = {'chi': '吃', 'pon': '碰', 'daiminkan': '杠'}[kind]
            values.append({'action': {'type': kind, 'pai': mjai(tile), 'consumed': sorted(mjai(t) for t in consumed)},
                           'name': label + ' ' + ' '.join(name(t) for t in (*consumed, tile)),
                           'tile': tile, 'sh': None, 'ukeire': None, 'safe': [], 'eligible': True})
    if 'hora' in offered and shanten(tuple(sorted(index(t) for t in [*board.hand, tile]))) == -1:
        values.append({'action': {'type': 'hora'}, 'name': '和牌', 'tile': tile, 'sh': None, 'ukeire': None, 'safe': [], 'eligible': True})
    return sorted(values, key=lambda v: json.dumps(v['action'], sort_keys=True))


class Adviser:
    def __init__(self):
        self.model = None
        self.device = None
        self.ready = False
        self.policy_name = 'Jev Discard v1'

    def prepare(self, checkpoint, progress=None):
        """Load and execute a real inference before reporting match readiness."""
        self.ready = False
        if progress:
            progress('Jev 加载中 · 请等待就绪后再匹配')
        description = self.load(checkpoint)
        if progress:
            progress('Jev 预热中 · 正在验证实际推理')
        self.recommend(Board(parse('123456m345p67s115z')), use_model=True)
        self.ready = True
        return description + ' · 已就绪'

    def load(self, checkpoint, device='cuda'):
        import torch
        from jevany import JevModel
        from jevany.checkpoint import LoadOptions
        from jevany.inference import InferenceOptions
        checkpoint = Path(checkpoint)
        from .model_identity import validate_checkpoint
        validate_checkpoint(checkpoint)
        manifest = checkpoint / 'manifest.json'
        if manifest.is_file():
            for filename, expected in json.loads(manifest.read_text(encoding='utf-8')).items():
                if hashlib.sha256((checkpoint / filename).read_bytes()).hexdigest() != expected:
                    raise ValueError('检查点文件校验失败：' + filename)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('未发现可用的CUDA显卡。请检查NVIDIA驱动和CUDA版PyTorch安装')
        dtype = 'bf16' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32'
        mirror = Path(os.environ.get('JEV_BASE_MODEL_DIR', checkpoint.parent.parent / 'base-model'))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        def load(path):
            return JevModel.from_pretrained(str(path), device=device, dtype=dtype,
                options=LoadOptions(merge=False, temperature=1., compile_mode=None, cuda_graphs=False,
                                    fused_kernels=False, base_load_path=str(mirror) if mirror.is_dir() else None),
                inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192,
                                                  max_packed_tokens=8192, prefix_cache_size=0))
        ordinary = load(checkpoint)
        call_checkpoint = checkpoint.parent / 'jev-call-v1'
        if call_checkpoint.is_dir():
            from .call_policy import CallPolicyClient, validate_call_checkpoint
            validate_call_checkpoint(call_checkpoint)
            calling = load(call_checkpoint)
            self.model = CallPolicyClient(ordinary, calling)
            self.policy_name = 'Jev Play v1'
        else:
            self.model = ordinary
            self.policy_name = 'Jev Discard v1'
        self.device = device
        return f'{self.policy_name} · {device.upper()} · {dtype}'

    def recommend(self, board, use_model=False):
        if board.reaction:
            return self.recommend_reaction(board, use_model)
        values = candidates(board)
        threats = set(board.riichi) - {board.seat}
        if use_model:
            if self.model is None:
                raise RuntimeError('Jev尚未加载，请先在设置中加载模型')
            from jevany.api import validate_distribution
            result = self.model.system_one(**request(board, values))
            keys = [f'a{i}' for i in range(len(values))]
            probabilities, _ = validate_distribution(result['answers']['action']['probabilities'], keys)
            for value, probability in zip(values, probabilities):
                value['preference'] = float(probability)
            ranked = sorted([v for v in values if v['eligible']], key=lambda v: -v['preference'])
        else:
            ranked = sorted([v for v in values if v['eligible'] and v['sh'] is not None], key=lambda v:
                (len(threats - set(v['safe'])) if threats and v['sh'] >= 1 else 0,
                 v['sh'], -v['ukeire'], v['dora'], v['red'], v['tile']))
        return {'top3': ranked[:3], 'all_candidates': values, 'engine': self.policy_name if use_model else '牌理引擎',
                'notes': board.notes(), 'winning_shape': shanten(tuple(sorted(index(t) for t in board.hand))) == -1,
                'riichi_selecting': board.riichi_selecting, 'strategy': hand_strategy(board, ranked[0]),
                'direction': hand_strategy(board, ranked[0], direction_only=True),
                'decision': decision_description(board, ranked[0])}

    def recommend_reaction(self, board, use_model=False):
        values = reaction_candidates(board)
        if len(values) < 2:
            raise ValueError('正在读取吃碰对应牌面')
        payload = request(board, [])
        payload['state']['screen_observation']['reaction'] = board.reaction
        payload['state']['screen_observation']['confirmed_metadata'] = ['honba', 'sticks']
        payload['questions']['action']['criteria'] = {f'a{i}': v['action'] for i, v in enumerate(values)}
        wins = [v for v in values if v['action']['type'] == 'hora']
        if wins:
            for value in values:
                value['preference'] = float(value is wins[0])
            ranked = sorted(values, key=lambda v: -v['preference'])
        elif use_model:
            if self.model is None:
                raise RuntimeError('Jev尚未加载')
            from jevany.api import validate_distribution
            output = self.model.system_one(**payload)
            probabilities, _ = validate_distribution(output['answers']['action']['probabilities'], list(payload['questions']['action']['criteria']))
            for value, probability in zip(values, probabilities):
                value['preference'] = float(probability)
            ranked = sorted(values, key=lambda v: -v['preference'])
        else:
            ranked = sorted(values, key=lambda v: 0 if v['action']['type'] == 'hora' else 1 if v['action']['type'] == 'none' else 2)
        return {'top3': ranked[:3], 'all_candidates': values, 'engine': self.policy_name if use_model else '牌理引擎',
                'notes': board.notes(), 'winning_shape': False, 'reaction': True,
                'strategy': hand_strategy(board, ranked[0]),
                'direction': hand_strategy(board, ranked[0], direction_only=True),
                'decision': decision_description(board, ranked[0])}
