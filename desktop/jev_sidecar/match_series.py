"""Read Mahjong Soul's end-of-match screens and count a requested series."""
from dataclasses import dataclass
import re
import time

import cv2
import numpy as np

from .autoplay import ClickPlan
from .screen_state import crop


CHEST_TITLE = (.648, .270, .080, .035)
CONFIRM_LABELS = ('确认', '確認')
REPLAY_LABELS = ('再来一场', '再來一場', '再莱一场')
CHEST_LABEL = re.compile(r'[铜銅银銀金]之[宝寶][匣画畫]')
RANK_METER = re.compile(r'\d{1,6}/\d{1,6}')
BUTTON_CLOSE = np.ones((5, 11), dtype=np.uint8)


@dataclass(frozen=True)
class MatchPage:
    kind: str
    action: ClickPlan


def button_boxes(image):
    width, height = 640, 400
    pixels = np.asarray(image.resize((width, height)), dtype=np.int16)
    red, green, blue = pixels.transpose(2, 0, 1)
    gold = (red > 175) & (green > 136) & (red - blue > 42)
    blue_button = (blue > 100) & (blue < 188) & (red > 40) & (blue - red > 37) & (blue - green > 20)
    matches = []
    for mask in (gold, blue_button):
        mask[:225] = False
        closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, BUTTON_CLOSE)
        _, _, stats, _ = cv2.connectedComponentsWithStats(closed)
        matches.append([(x / width, y / height, w / width, h / height) for x, y, w, h, area in stats[1:]
                        if 50 <= w <= 145 and 15 <= h <= 47 and area >= .55 * w * h])
    return matches


def label(image, rect, ocr):
    area = crop(image, rect)
    if area.height < 48:
        area = area.resize((round(area.width * 48 / max(area.height, 1)), 48))
    text, _ = ocr(np.asarray(area), use_det=False, use_cls=False)
    return ''.join(word for word, score in text or [] if score >= .62).replace(' ', '')


def match_page(kind, rect, text):
    x, y, w, h = rect
    return MatchPage(kind, ClickPlan(x + w / 2, y + h / 2, text))


def paired_buttons(left, right):
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    return (abs(ly + lh / 2 - ry - rh / 2) <= .028
            and .67 <= lw / rw <= 1.5 and .68 <= lh / rh <= 1.48
            and -.006 <= rx - lx - lw <= .10)


def page_text(image, rect, ocr):
    rows, _ = ocr(np.asarray(crop(image, rect)), use_cls=False)
    return [word.replace(' ', '') for _, word, score in rows or [] if score >= .66]


def read_match_page(image, ocr_engine):
    """Locate the current action from visible controls and contextual text."""
    image = image.convert('RGB')
    gold, blue = button_boxes(image)
    if not gold:
        return None
    ocr = ocr_engine()
    modal = [(ok, cancel) for ok in gold if .20 <= ok[0] <= .42 and .59 <= ok[1] <= .81
             for cancel in blue if .43 <= cancel[0] <= .64 and paired_buttons(ok, cancel)]
    if modal:
        lines = page_text(image, (.24, .10, .52, .48), ocr)
        heading = any(line in REPLAY_LABELS or line in ('要再进行一场', '要再進行一場') for line in lines)
        context = any('匹配对局' in line or '匹配對局' in line for line in lines)
        if heading and context:
            for ok, cancel in modal:
                if label(image, ok, ocr) in CONFIRM_LABELS and label(image, cancel, ocr) == '取消':
                    return match_page('confirm', ok, '确认下一场匹配')
        # A visible modal takes precedence over the settlement underneath.
        return None

    confirm = next((box for box in gold if box[0] >= .80 and box[1] >= .78
                    and label(image, box, ocr) in CONFIRM_LABELS), None)
    if confirm is None:
        return None
    for replay in blue:
        if replay[0] >= .63 and paired_buttons(replay, confirm) and label(image, replay, ocr) in REPLAY_LABELS:
            return match_page('rewards', replay, '再来一场')

    title_boxes = ((.427, .035, .122, .080), (.418, 0, .159, .074))
    if not any(label(image, box, ocr) in ('终局', '終局') for box in title_boxes):
        return None
    meter = label(image, (.820, .455, .120, .070), ocr)
    if RANK_METER.fullmatch(meter):
        return match_page('rank', confirm, '确认段位进度')
    # The stylized 匣 can be read as 画 by RapidOCR. This title is checked
    # only at its fixed spot on a confirmed end-of-match screen.
    chest = label(image, CHEST_TITLE, ocr)
    if CHEST_LABEL.fullmatch(chest):
        return match_page('chest', confirm, '确认宝匣与羁绊进度')
    progress = page_text(image, (.61, .18, .38, .44), ocr)
    if any(RANK_METER.fullmatch(line) for line in progress):
        return match_page('rank', confirm, '确认段位进度')
    if any(CHEST_LABEL.fullmatch(line.strip('—-')) for line in progress):
        return match_page('chest', confirm, '确认宝匣与羁绊进度')
    return match_page('results', confirm, '确认终局成绩')


def shows_table(image):
    area = np.asarray(crop(image.convert('RGB'), (.466, .374, .068, .073)), dtype=np.int16)
    if not area.size:
        return False
    r, g, b = area.transpose(2, 0, 1)
    # The center's cyan round/tile display dims under the terminal and
    # matching overlays even when remnants of an old hand are still visible.
    return float(((g > 115) & (b > 100) & (g - r > 25) & (b - r > 25) & (g > b - 38)).mean()) > .045


class MatchSeries:
    """Count each played match once and act on any recognized settlement screen."""
    def __init__(self):
        self.enabled = False
        self.total = self.completed = self.revision = 0
        self.phase = 'waiting'
        self.previous = None
        self.clicks = {}

    def start(self, total):
        if total < 1:
            raise ValueError('连打场数至少为 1')
        self.revision += 1
        self.enabled = True
        self.total = total
        self.completed = 0
        self.phase = 'waiting'
        self.previous = None
        self.clicks.clear()

    def stop(self):
        self.revision += 1
        self.enabled = False

    def observe(self, page=None, playing=False, *, now=None):
        if not self.enabled or self.phase == 'complete':
            return None
        current = page.kind if page else 'board' if playing else None
        previous, previous_action = self.previous or (None, None)
        action = page.action if page else None
        stable = current is not None and current == previous
        if stable and action and previous_action:
            stable = abs(action.x - previous_action.x) <= .025 and abs(action.y - previous_action.y) <= .025
        self.previous = current, action
        if not stable:
            return None
        if current == 'board':
            if self.phase != 'playing':
                self.phase = 'playing'
                self.clicks.clear()
            return None
        if self.phase == 'playing':
            self.completed += 1
            self.phase = 'complete' if self.completed >= self.total else 'settlement'
        elif self.phase == 'waiting':
            # Resume from a visible settlement without counting the previous match.
            self.phase = 'settlement'
        if self.phase not in ('settlement', 'queued'):
            return None
        attempts, last = self.clicks.get(current, (0, 0))
        now = time.monotonic() if now is None else now
        if attempts >= 2 or attempts and now - last < 5:
            return None
        return action

    def clicked(self, page, *, now=None):
        now = time.monotonic() if now is None else now
        attempts, _ = self.clicks.get(page.kind, (0, 0))
        self.clicks[page.kind] = attempts + 1, now
        if page.kind == 'confirm':
            self.phase = 'queued'
