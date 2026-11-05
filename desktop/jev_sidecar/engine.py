"""Discard features and the trained Jev interface for screenshot-derived state."""
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import json
import re
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
    threats = set(board.riichi) - {board.seat}
    fold = bool(threats) and min(v['sh'] for v in values) >= 1
    ref = min(values, key=lambda v: (len(threats - set(v['safe'])) if fold else 0, v['sh'], -v['ukeire'], v['red'], tiles.index(v['tile'])))
    eligible = [v for v in values if v['sh'] <= ref['sh'] and
                (not fold or len(threats - set(v['safe'])) <= len(threats - set(ref['safe'])))]
    for value in values:
        value['eligible'] = value in eligible
    return values


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
    criteria = {f'a{i}': {'type': 'dahai', 'pai': mjai(v['tile']),
                         **{key: v[key] for key in ['sh', 'ukeire', 'safe', 'red']}} for i, v in enumerate(values)}
    return {'state': state, 'questions': {'action': {'type': 'choice', 'instructions': INSTRUCTION, 'criteria': criteria}}}


class Adviser:
    def __init__(self):
        self.model = None
        self.device = None

    def load(self, checkpoint, device='cuda'):
        import torch
        from jevany import JevModel
        from jevany.checkpoint import LoadOptions
        from jevany.inference import InferenceOptions
        checkpoint = Path(checkpoint)
        if hashlib.sha256((checkpoint / 'head.pt').read_bytes()).hexdigest() != CHECKPOINT_SHA:
            raise ValueError('请选择本工具附带的已确认校正版2；权重指纹不匹配')
        manifest = checkpoint / 'manifest.json'
        if manifest.is_file():
            for filename, expected in json.loads(manifest.read_text(encoding='utf-8')).items():
                if hashlib.sha256((checkpoint / filename).read_bytes()).hexdigest() != expected:
                    raise ValueError('检查点文件校验失败：' + filename)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('未发现可用的CUDA显卡。请检查NVIDIA驱动和CUDA版PyTorch安装')
        dtype = 'bf16' if device == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32'
        mirror = checkpoint.parent.parent / 'base-model'
        self.model = JevModel.from_pretrained(str(checkpoint), device=device, dtype=dtype,
            options=LoadOptions(merge=False, temperature=1., compile_mode=None, cuda_graphs=False,
                                fused_kernels=False, base_load_path=str(mirror) if mirror.is_dir() else None),
            inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192, max_packed_tokens=8192))
        self.device = device
        return f'Jev 校正版2 · {device.upper()} · {dtype}'

    def recommend(self, board, use_model=False):
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
            ranked = sorted([v for v in values if v['eligible']], key=lambda v:
                (len(threats - set(v['safe'])) if threats and v['sh'] >= 1 else 0,
                 v['sh'], -v['ukeire'], v['dora'], v['red'], v['tile']))
        return {'top3': ranked[:3], 'all_candidates': values, 'engine': 'Jev 校正版2' if use_model else '牌理引擎',
                'notes': board.notes(), 'winning_shape': shanten(tuple(sorted(index(t) for t in board.hand))) == -1}
