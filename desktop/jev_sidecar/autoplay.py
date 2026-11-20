"""Execute one reviewed recommendation against the foreground game window."""
from dataclasses import dataclass
from collections import Counter
import ctypes
from ctypes import wintypes
import time

from .tiles import mjai
from .screen_state import crop


@dataclass(frozen=True)
class ClickPlan:
    x: float
    y: float
    label: str


class RiichiFlow:
    """Keep Jev's discard choice through the separate riichi selection UI."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.hand = None
        self.tile = None
        self.slots = ()
        self.started = 0.
        self.clicked = False

    def begin(self, hand, tile, detections=()):
        self.hand, self.tile = Counter(hand), tile
        self.slots = tuple(detections)
        self.started, self.clicked = time.monotonic(), False

    def plan(self, detections, screen, hand_rect, image_size):
        if self.hand is None:
            return None
        current = Counter(d.tile for d in detections)
        selecting = screen.get('riichi_selecting')
        if time.monotonic() - self.started > 25 or (
            not selecting and len(detections) >= 13 and sum((current - self.hand).values()) > 2
        ):
            self.reset()
            return None
        if not selecting:
            if screen.get('own_riichi') and self.hand - current == Counter([self.tile]) and not current - self.hand:
                self.reset()
            return None
        if self.clicked:
            return None
        if not current - self.hand and sum((self.hand - current).values()) <= 1:
            return tile_click(self.tile, detections, hand_rect, image_size)
        # Greyed-out tiles can all be misclassified during selection. Require
        # the chosen tile itself to match its slot and identity before pressing riichi.
        if len(self.slots) != 14:
            return None
        aligned = {}
        for index, original in enumerate(self.slots):
            center = (original.box[0] + original.box[2]) / 2
            tolerance = (original.box[2] - original.box[0]) * .24
            matches = [d for d in detections if abs((d.box[0] + d.box[2]) / 2 - center) <= tolerance]
            if len(matches) == 1:
                aligned[index] = matches[0]
        if len(aligned) < 11:
            return None
        choices = [aligned[i] for i, original in enumerate(self.slots)
                   if original.tile == self.tile and i in aligned and aligned[i].tile == self.tile]
        return tile_click(self.tile, choices, hand_rect, image_size)


def tile_click(tile, detections, hand_rect, image_size):
    matches = [d for d in detections if d.tile == tile]
    if not matches:
        return None
    d = matches[-1]
    x, y, _, _ = hand_rect
    width, height = image_size
    return ClickPlan(x + (d.box[0] + d.box[2]) / 2 / width,
                     y + (d.box[1] + (d.box[3] - d.box[1]) * .40) / height,
                     '打 ' + tile)


def plan_click(result, detections, screen, hand_rect, image_size):
    top = result['top3'][0]
    action = top.get('action', {}).get('type', 'dahai')
    offered = screen.get('offered', [])
    if 'tsumo' in offered:
        return ClickPlan(.53, .752, '自摸') if action == 'hora' else None
    if action == 'reach':
        return ClickPlan(.53, .752, '立直') if screen.get('riichi_available') else None
    if action == 'none':
        return ClickPlan(.683, .752, '跳过') if screen.get('reaction') and offered else None
    if action in ('ankan', 'kakan'):
        options = [v for v in result['all_candidates'] if v.get('action', {}).get('type') in ('ankan', 'kakan')]
        if not screen.get('self_kan_available') or 'daiminkan' not in offered or len(options) != 1:
            return None
        return ClickPlan(.53 - .13 * (len(offered) - 1 - offered.index('daiminkan')), .752, top['name'])
    if action in ('hora', 'chi', 'pon', 'daiminkan'):
        if action not in offered or (action != 'hora' and not screen.get('reaction')):
            return None
        if action in ('chi', 'pon', 'daiminkan'):
            options = [v for v in result['all_candidates'] if v.get('action', {}).get('type') == action]
            if len(options) != 1 and action not in ('chi', 'pon'):
                return None
        return ClickPlan(.53 - .13 * (len(offered) - 1 - offered.index(action)), .752, top['name'])
    if len(detections) not in (2, 5, 8, 11, 14) or screen.get('reaction'):
        return None
    if len(screen.get('melds') or []) != (14 - len(detections)) // 3:
        return None
    # Prefer the separated draw when the same face occurs more than once.
    return tile_click(top.get('tile'), detections, hand_rect, image_size)


def call_options_click(image, detector, options, selected, label):
    """Match the two face-up tiles for each choice in the call selector."""
    rect = (.17, .62, .66, .19)
    area = crop(image, rect)
    tiles = [d for d in detector.detect(area, confidence=.35)
             if .025 * image.width <= d.box[2] - d.box[0] <= .061 * image.width
             and .06 * image.height <= d.box[3] - d.box[1] <= .15 * image.height]
    tiles.sort(key=lambda d: d.box[0])
    if len(tiles) != 2 * len(options):
        return None
    pairs = list(zip(tiles[::2], tiles[1::2]))
    if any(abs(left.box[1] - right.box[1]) > .025 * image.height
           or not -.01 * image.width <= right.box[0] - left.box[2] <= .02 * image.width
           for left, right in pairs):
        return None
    keys = [tuple(sorted((mjai(left.tile), mjai(right.tile)))) for left, right in pairs]
    if set(keys) != options or keys.count(selected) != 1:
        return None
    tile = pairs[keys.index(selected)][0]
    x1, y1, x2, y2 = tile.box
    return ClickPlan(rect[0] + (x1 + x2) / (2 * image.width),
                     rect[1] + (y1 + y2) / (2 * image.height), label)


class AutoPlayer:
    def __init__(self, clicker=None):
        self.clicker = clicker or click_game
        self.executed = None
        self.last_click = 0.
        self.pending = None

    def reset(self):
        self.executed = None
        self.pending = None

    def observe(self, hand, screen):
        """A sent input is pending until a later frame shows the action ended."""
        if self.pending is None:
            return None
        before, old_screen, label = self.pending
        if tuple(hand) == before and screen == old_screen:
            return '等待游戏响应：' + label
        self.pending = None
        return '点击后画面已变化：' + label

    def execute(self, key, plan, capture, table_rect=None):
        if plan is None or self.pending is not None or key == self.executed or time.monotonic() - self.last_click < .6:
            return False
        self.clicker(capture, plan.x, plan.y, table_rect)
        self.executed = key
        self.pending = (tuple(key[1]), key[2], plan.label) if isinstance(key, tuple) else None
        self.last_click = time.monotonic()
        return True


def escape_pressed():
    return bool(ctypes.WinDLL('user32').GetAsyncKeyState(0x1B) & 0x8000)


def click_game(capture, x, y, table_rect=None):
    user = ctypes.WinDLL('user32', use_last_error=True)
    user.GetForegroundWindow.restype = wintypes.HWND
    if int(user.GetForegroundWindow() or 0) != capture.hwnd:
        raise RuntimeError('自动点击等待雀魂回到前台；后台仍继续识别')
    if escape_pressed():
        raise RuntimeError('Esc 已停止自动点击')
    if not capture.geometry or time.monotonic() - (capture.frame_time or 0) > 2:
        raise RuntimeError('等待新的雀魂画面后再点击')
    raw_width, raw_height, bounds = capture.geometry
    rect = wintypes.RECT()
    user.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    if not user.GetClientRect(capture.hwnd, ctypes.byref(rect)):
        raise RuntimeError('雀魂窗口位置不可用')
    if abs(rect.right - raw_width) > 2 or abs(rect.bottom - raw_height) > 2:
        raise RuntimeError('自动点击请使用全屏或无边框雀魂窗口')
    origin = wintypes.POINT(0, 0)
    user.ClientToScreen(capture.hwnd, ctypes.byref(origin))
    tx, ty, tw, th = table_rect or (0, 0, 1, 1)
    left, top, right, bottom = bounds
    px = origin.x + round(left + (tx + x * tw) * (right - left))
    py = origin.y + round(top + (ty + y * th) * (bottom - top))
    point = wintypes.POINT(px, py)
    user.WindowFromPoint.argtypes = [wintypes.POINT]
    user.WindowFromPoint.restype = wintypes.HWND
    user.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user.GetAncestor.restype = wintypes.HWND
    hit = user.WindowFromPoint(point)
    if int(user.GetAncestor(hit, 2) or hit or 0) != capture.hwnd:
        raise RuntimeError('推荐位置被其他窗口遮挡，自动点击暂缓')
    class Mouse(ctypes.Structure):
        _fields_ = [('dx', wintypes.LONG), ('dy', wintypes.LONG), ('mouseData', wintypes.DWORD),
                    ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD), ('dwExtraInfo', ctypes.c_size_t)]
    class Payload(ctypes.Union):
        _fields_ = [('mi', Mouse)]
    class Input(ctypes.Structure):
        _fields_ = [('type', wintypes.DWORD), ('data', Payload)]
    user.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
    user.SendInput.restype = wintypes.UINT
    # SetCursorPos uses physical pixels in this DPI-aware Qt application.
    if not user.SetCursorPos(px, py):
        raise RuntimeError('无法定位鼠标')
    # Unity can poll mouse state once per frame; a down/up in the same batch
    # can disappear between frames. Let hover settle and hold across frames.
    time.sleep(.05)
    if int(user.GetForegroundWindow() or 0) != capture.hwnd or escape_pressed():
        raise RuntimeError('窗口已切换或 Esc 已停止，取消点击')
    down, up = Input(), Input()
    down.data.mi.dwFlags = 0x0002
    up.data.mi.dwFlags = 0x0004
    if user.SendInput(1, ctypes.byref(down), ctypes.sizeof(Input)) != 1:
        raise RuntimeError('无法发送鼠标按下')
    try:
        time.sleep(.07)
    finally:
        if user.SendInput(1, ctypes.byref(up), ctypes.sizeof(Input)) != 1:
            raise RuntimeError('无法发送鼠标抬起')
