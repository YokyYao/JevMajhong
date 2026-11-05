"""Portable tile notation. No game-client integration is required."""
from collections import Counter
import re

HONORS = dict(zip('ESWNPFC', ['1z', '2z', '3z', '4z', '5z', '6z', '7z']))
HONORS.update(dict(zip('東南西北白發中', ['1z', '2z', '3z', '4z', '5z', '6z', '7z'])))
HONORS.update({'东': '1z', '发': '6z'})


def index(tile):
    return 'mpsz'.index(tile[1]) * 9 + (4 if tile[0] == '0' else int(tile[0]) - 1)


def normal(tile):
    return '5' + tile[1] if tile[0] == '0' else tile


def parse(text):
    text = re.sub(r'[\s,，]+', '', text)
    result, pos = [], 0
    while pos < len(text):
        match = re.match(r'([0-9]+)([mpsz])', text[pos:])
        if match:
            digits, suit = match.groups()
            for digit in digits:
                if (suit == 'z' and digit not in '1234567') or (suit != 'z' and digit not in '0123456789'):
                    raise ValueError('牌代码应为1m–9s、1z–7z，赤五用0m/0p/0s')
                result.append(digit + suit)
            pos += len(match.group())
        elif text[pos] in HONORS:
            result.append(HONORS[text[pos]])
            pos += 1
        else:
            raise ValueError('无法读取牌代码：' + text[pos:pos + 12])
    counts = Counter(normal(t) for t in result)
    if any(n > 4 for n in counts.values()):
        raise ValueError('同一种牌超过4张，请校正识别结果')
    if any(result.count('0' + s) > 1 for s in 'mps'):
        raise ValueError('同一种赤五超过1张，请校正识别结果')
    return result


def name(tile):
    if tile[1] == 'z':
        return ['东', '南', '西', '北', '白', '发', '中'][int(tile[0]) - 1]
    return ('赤五' if tile[0] == '0' else '一二三四五六七八九'[int(tile[0]) - 1]) + {'m': '万', 'p': '筒', 's': '索'}[tile[1]]


def mjai(tile):
    if tile[1] == 'z':
        return 'ESWNPFC'[int(tile[0]) - 1]
    return '5' + tile[1] + 'r' if tile[0] == '0' else tile


def dora(indicator):
    i = index(indicator)
    if i < 27:
        return i // 9 * 9 + (i + 1) % 9
    if i < 31:
        return 27 + (i - 27 + 1) % 4
    return 31 + (i - 31 + 1) % 3
