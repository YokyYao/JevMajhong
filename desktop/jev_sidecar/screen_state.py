"""Read visible call buttons and the player's open melds from table pixels."""
from functools import lru_cache
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from .engine import Board, parse_melds, reaction_candidates
from .tiles import mjai, normal
from .table_counters import TableCounterReader


def crop(image, rect):
    x, y, w, h = rect
    return image.crop((round(x * image.width), round(y * image.height),
                       round((x + w) * image.width), round((y + h) * image.height)))


def split_melds(tiles, count):
    @lru_cache(None)
    def split(start, remaining):
        if remaining == 0:
            return () if start == len(tiles) else None
        for size in (3, 4):
            group = tiles[start:start + size]
            if len(group) != size:
                continue
            try:
                parse_melds(''.join(group))
            except ValueError:
                continue
            tail = split(start + size, remaining - 1)
            if tail is not None:
                return (''.join(group),) + tail
        return None
    groups = split(0, count)
    return parse_melds(','.join(groups)) if groups is not None else None


def riichi_selection_overlay(image):
    """Gold undo control shown only while choosing the riichi discard."""
    # The countdown starts just to the right. Including it makes yellow
    # 4/3/2/1 digits look like this gold control in fast ranked games.
    area = np.asarray(crop(image.convert('RGB'), (.725, .73, .05, .09))).astype(float)
    gold = (area[:, :, 0] > 150) & (area[:, :, 1] > 110) & (area[:, :, 1] < area[:, :, 0] * .95) & (area[:, :, 2] < area[:, :, 1] * .7)
    edge = max(1, gold.shape[1] // 5)
    return float(gold.mean()) > .035 and gold[:, :edge].mean() > .04 and gold[:, -edge:].mean() > .04


class ScreenStateReader:
    def __init__(self, state_path=None):
        self.ocr = None
        self.counter_reader = TableCounterReader()
        self.rivers = {}
        self.last_discard = None
        self.previous_hand = []
        self.previous_reaction = None
        self.melds = []
        self.forbidden = []
        self.riichi_selecting = False
        self.own_riichi = False
        self.hand_id = 0
        self.expected_riichi_hand = None
        self.expected_riichi_tile = None
        self.state_path = Path(state_path) if state_path else None
        if self.state_path and self.state_path.is_file():
            try:
                saved = json.loads(self.state_path.read_text(encoding='utf-8'))
                if time.time() - saved['updated'] < 3600:
                    self.previous_hand = saved['hand']
                    self.previous_reaction = saved['reaction']
                    self.melds = saved['melds']
                    self.forbidden = saved.get('forbidden', [])
            except (ValueError, KeyError, OSError):
                pass

    def ocr_engine(self):
        if self.ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            self.ocr = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1,
                                det_limit_type='max', det_limit_side_len=640)
        return self.ocr

    def buttons(self, image):
        if image.width > 1280:
            image = image.resize((1280, round(image.height * 1280 / image.width)))
        self.ocr_engine()
        area = crop(image, (.35, .57, .64, .27))
        if area.width > 640:
            area = area.resize((640, round(area.height * 640 / area.width)))
        output, _ = self.ocr(np.asarray(area), use_cls=False)
        labels = ''.join(item[1] for item in output or [])
        skip_image = crop(image, (.63, .70, .13, .095))
        skip_text, _ = self.ocr(np.asarray(skip_image), use_det=False, use_cls=False)
        labels += ''.join(item[0] for item in skip_text or [])
        # A skip button distinguishes a live reaction overlay from table labels.
        if not any(word in labels for word in ('跳过', '跳過', '取消')):
            return []
        # Stylized green 吃 is sometimes omitted by text-box detection. Read
        # action label slots directly once the skip button proves an overlay.
        for center in (.53, .40, .27):
            label_image = crop(image, (center - .04, .70, .08, .095))
            text, _ = self.ocr(np.asarray(label_image), use_det=False, use_cls=False)
            labels += ''.join(item[0] for item in text or [])
        # The orange calligraphy 立 is read as 红 by RapidOCR on this skin.
        labels = labels.replace('红直', '立直').replace('紅直', '立直')
        return (['reach_select'] if '取消' in labels and '立直' not in labels and riichi_selection_overlay(image) else []) + [kind for text, kind in [('吃', 'chi'), ('碰', 'pon'), ('杠', 'daiminkan'),
                                       ('槓', 'daiminkan'), ('和', 'hora'), ('胡', 'hora'), ('立直', 'reach')]
                if text in labels]

    def read(self, image, regions, detector, found):
        hand = [d.tile for d in found.get('hand', [])]
        meld_count = (14 - len(hand)) // 3 if len(hand) % 3 == 2 else (13 - len(hand)) // 3
        if self.previous_reaction and meld_count == len(self.melds) + 1:
            consumed = Counter(self.previous_hand) - Counter(hand)
            for candidate in reaction_candidates(Board(self.previous_hand, reaction=self.previous_reaction)):
                action = candidate['action']
                # Compare portable tile codes, preserving red fives.
                removed = Counter(mjai(t) for t in consumed.elements())
                if action['type'] in ('chi', 'pon', 'daiminkan') and removed == Counter(action['consumed']):
                    tiles = list(consumed.elements()) + [self.previous_reaction['tile']]
                    self.melds += parse_melds(''.join(tiles))
                    called = normal(self.previous_reaction['tile'])
                    self.forbidden = [called]
                    used = sorted(int(normal(t)[0]) for t in consumed.elements())
                    if action['type'] == 'chi' and used[1] - used[0] == 1:
                        self.forbidden = [str(n) + called[1] for n in (used[0] - 1, used[1] + 1) if 1 <= n <= 9]
                    break
        if len(hand) % 3 == 1:
            self.forbidden = []
        melds = None
        if 0 < meld_count <= 4:
            if len(self.melds) == meld_count:
                melds = self.melds
            else:
                region = regions.get('meld0', {'rect': [.77, .79, .23, .20]})
                detected = detector.detect(crop(image, region['rect']))
                detected.sort(key=lambda d: d.box[0])
                melds = split_melds([d.tile for d in detected], meld_count)
        elif meld_count == 0:
            melds = []
            self.melds = []
            self.forbidden = []
        if melds is not None:
            self.melds = melds
        changed = []
        for seat in range(4):
            values = tuple(d.tile for d in found.get(f'river{seat}', []))
            previous = self.rivers.get(seat)
            if previous is not None and len(values) > len(previous) and values[:len(previous)] == previous:
                changed.append((seat, values[-1]))
            self.rivers[seat] = values
        if len(changed) == 1:
            self.last_discard = changed[0]
        if not any(self.rivers.values()):
            self.last_discard = None
        offered = self.buttons(image) if len(hand) in (1, 2, 4, 5, 7, 8, 10, 11, 13, 14) else []
        # A new hand clears the previous riichi lock. Own river shrinks at a deal.
        own_river = tuple(d.tile for d in found.get('river0', []))
        # Settlement animations can expose only six tiles and overwrite
        # previous_hand. Compare complete closed hands across those frames.
        reference = getattr(self, '_last_closed_hand', self.previous_hand)
        replaced = sum((Counter(reference) - Counter(hand)).values())
        if len(hand) in (13, 14) and len(reference) in (13, 14) and len(own_river) <= 1 and replaced >= 6:
            self.hand_id += 1
            self.own_riichi = False
            self.riichi_selecting = False
            self._own_river_max = 0
            self.expected_riichi_hand = None
            self.expected_riichi_tile = None
        if len(hand) in (13, 14):
            self._last_closed_hand = list(hand)
        self._own_river_max = max(getattr(self, '_own_river_max', 0), len(own_river))
        available = 'reach' in offered and len(hand) == 14 and not self.own_riichi
        was_selecting = self.riichi_selecting
        selecting = len(hand) in (12, 13, 14) and not available and riichi_selection_overlay(image)
        if self.riichi_selecting and len(hand) == 13 and not selecting:
            original = self.expected_riichi_hand or self.previous_hand
            removed = Counter(original) - Counter(hand)
            expected = Counter([self.expected_riichi_tile]) if self.expected_riichi_tile else removed
            if len(original) == 14 and sum(removed.values()) == 1 and removed == expected and not Counter(hand) - Counter(original):
                self.own_riichi = True
                self.riichi_selecting = False
        # After pressing reach, the next frame still has fourteen tiles but
        # the action buttons disappear while the game asks for a tenpai discard.
        if len(hand) in (12, 13, 14):
            pending_incomplete = was_selecting and len(hand) == 13 and self.expected_riichi_tile is not None
            self.riichi_selecting = bool(selecting or pending_incomplete) and not self.own_riichi
        self._reach_available = available
        calls = [kind for kind in offered if kind not in ('reach', 'reach_select')] if len(hand) % 3 == 1 else []
        reaction = None
        if calls:
            offered = calls
            reaction = self.highlighted_reaction(image, regions, detector, found, offered)
            possible = [self.last_discard] if self.last_discard else []
            possible += [(seat, values[-1]) for seat, values in self.rivers.items() if seat != 0 and values]
            valid = {}
            for seat, tile in possible:
                if seat == 0:
                    continue
                event = {'seat': seat, 'tile': tile, 'offered': offered}
                board = Board(hand, reaction=event)
                try:
                    if set(offered) <= {v['action']['type'] for v in reaction_candidates(board)}:
                        valid[(seat, tile)] = event
                except ValueError:
                    continue
                if reaction is None and self.last_discard == (seat, tile) and (seat, tile) in valid:
                    reaction = event
                    break
            if reaction is None and len(valid) == 1:
                reaction = next(iter(valid.values()))
            if reaction is None and 'chi' in offered and 'pon' in offered:
                # Both buttons constrain the called face to an owned pair
                # that also forms a sequence, often uniquely identifying it.
                possible_calls = []
                for tile in set(normal(t) for t in hand):
                    event = {'seat': 3, 'tile': tile, 'offered': offered}
                    if set(offered) <= {v['action']['type'] for v in reaction_candidates(Board(hand, reaction=event))}:
                        possible_calls.append(event)
                if len(possible_calls) == 1:
                    reaction = possible_calls[0]
        # Buttons disappear before the call animation removes consumed tiles.
        # Keep that event through unchanged intervening frames.
        self.previous_reaction = reaction or (self.previous_reaction if hand == self.previous_hand else None)
        self.previous_hand = hand
        if self.state_path and melds is not None and hand:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.state_path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'updated': time.time(), 'hand': hand, 'reaction': self.previous_reaction, 'melds': melds,
                                             'forbidden': self.forbidden}), encoding='utf-8')
            temporary.replace(self.state_path)
        return {'melds': melds, 'offered': offered, 'reaction': reaction, 'forbidden': self.forbidden, 'hand_id': self.hand_id,
                'riichi_available': available, 'riichi_selecting': self.riichi_selecting,
                'own_riichi': self.own_riichi,
                'counters': self.counter_reader.read(image, regions.get('counters'), self.ocr_engine)}

    def highlighted_reaction(self, image, regions, detector, found, offered):
        """The green last-discard marker identifies the tile the buttons refer to."""
        hand = [d.tile for d in found.get('hand', [])]
        choices = []
        for seat in (1, 2, 3):
            allowed = []
            for suit in 'mpsz':
                for number in range(1, 8 if suit == 'z' else 10):
                    tile = str(number) + suit
                    event = {'seat': seat, 'tile': tile, 'offered': offered}
                    if set(offered) <= {v['action']['type'] for v in reaction_candidates(Board(hand, reaction=event))}:
                        allowed.append(tile)
            if not allowed or f'river{seat}' not in regions:
                continue
            region = regions[f'river{seat}']
            area = crop(image, region['rect']).rotate(int(region.get('rotation', 0)), expand=True)
            array = np.asarray(area).astype(float)
            green = (array[:, :, 1] > 180) & (array[:, :, 1] > array[:, :, 0] * 1.2) & (array[:, :, 1] > array[:, :, 2] * 1.1)
            for detection in detector.detect(area, confidence=.01, allowed_tiles=allowed):
                x1, y1, x2, y2 = [int(v) for v in detection.box]
                pad = max(8, (y2 - y1) // 3)
                border = green.copy()
                border[max(0, y1):y2, max(0, x1):x2] = False
                marker = int(border[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad].sum())
                if marker < 12:
                    continue
                tile = detection.tile
                # Sideways river faces have perspective distortion. Comparing
                # their vectors to upright hand tiles can override a correct
                # detector label with an unrelated pair (East -> six circles).
                choices.append((marker, {'seat': seat, 'tile': tile, 'offered': offered}))
        return max(choices, key=lambda value: value[0])[1] if choices else None
