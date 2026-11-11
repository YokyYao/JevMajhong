"""Always-on-top Windows side panel with calibrated local screen capture."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QObject, QRunnable, QRectF, Qt, QThreadPool, QTimer, Signal, Slot
from PySide6.QtGui import QFontDatabase, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QFrame, QGraphicsScene, QGraphicsView, QGridLayout, QHBoxLayout, QInputDialog, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget, QHeaderView, QSizePolicy)

from .engine import Adviser, Board, parse_melds
from .tiles import name, parse
from .vision import Templates, TileDetector, hand_boxes
from .screen_state import ScreenStateReader
from .window_capture import CaptureUnavailable, GameWindowCapture
from .dataset import DecisionRecorder
from .autoplay import AutoPlayer, RiichiFlow, plan_click, escape_pressed

ROOT = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[2]
CONFIG = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.config')) / 'JevMahjong' / 'settings.json'
DEMO_HAND = '123456m345p67s115z'
ZONE_NAMES = {'hand': '自己的手牌', 'river0': '自己的牌河', 'river1': '下家牌河', 'river2': '对家牌河',
              'river3': '上家牌河', 'dora': '宝牌指示牌', 'meld0': '自己的副露',
              'counters': '立直棒与本场（整行）'}
DEFAULT_RIVERS = {
    'river0': {'rect': [.39, .505, .23, .20], 'rotation': 0},
    'river1': {'rect': [.595, .22, .14, .295], 'rotation': 90},
    'river2': {'rect': [.39, .13, .20, .165], 'rotation': 180},
    'river3': {'rect': [.265, .275, .14, .285], 'rotation': 270},
}


def include_drawn_tile(region):
    if region.get('include_drawn'):
        return region
    x, y, w, h = region['rect']
    top = max(0, y - h * .65)
    right = min(1, x + w + w * 2 / 13)
    return {**region, 'rect': [x, top, right - x, min(1, y + h) - top], 'include_drawn': True}
STYLE = """
QWidget { background:#101923; color:#e9f1fa; font-family:'Microsoft YaHei UI','Noto Sans SC','Noto Sans CJK SC','Arial'; font-size:13px; }
QMainWindow { background:#101923; } QFrame#card { background:#182535; border:1px solid #2b3b50; border-radius:12px; }
QLabel { background:transparent; } QLabel#muted { color:#9bafc6; } QLabel#hero { color:#63e4bf; font-size:44px; font-weight:700; }
QPushButton { background:#24364b; border:1px solid #3a4d64; padding:8px 10px; border-radius:7px; }
QPushButton:hover { background:#314c65; } QPushButton#primary { background:#207960; border-color:#4fbc96; font-weight:600; }
QPushButton:disabled { color:#667689; } QLineEdit,QTextEdit,QSpinBox,QComboBox { background:#0c1420; border:1px solid #35455b; padding:6px; border-radius:5px; }
QTabWidget::pane { border:0; } QTabBar::tab { background:#182535; padding:10px 15px; } QTabBar::tab:selected { color:#63e4bf; }
QTableWidget { background:#182535; border:0; gridline-color:#2b3b50; } QHeaderView::section { background:#24364b; color:#adc0d4; border:0; padding:6px; }
QCheckBox { spacing:7px; } QScrollArea { border:0; }
"""


def pixmap(image):
    return QPixmap.fromImage(ImageQt(image.convert('RGBA')))


def pillow(image):
    image = image.convertToFormat(QImage.Format.Format_RGB888)
    return Image.frombytes('RGB', (image.width(), image.height()), bytes(image.bits()), 'raw', 'RGB', image.bytesPerLine()).copy()


class RegionView(QGraphicsView):
    def __init__(self, image):
        super().__init__()
        self.image = image
        self.setScene(QGraphicsScene(self))
        self.scene().addPixmap(pixmap(image))
        self.scene().setSceneRect(0, 0, image.width, image.height)
        self.origin = self.rect_item = self.selection = None
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event):
        self.origin = self.mapToScene(event.position().toPoint())
        if self.rect_item:
            self.scene().removeItem(self.rect_item)
        self.rect_item = self.scene().addRect(QRectF(self.origin, self.origin), QPen(Qt.GlobalColor.green, 3))

    def mouseMoveEvent(self, event):
        if self.origin is not None:
            rect = QRectF(self.origin, self.mapToScene(event.position().toPoint())).normalized().intersected(self.sceneRect())
            self.rect_item.setRect(rect)
            self.selection = rect

    def mouseReleaseEvent(self, event):
        self.mouseMoveEvent(event)
        self.origin = None


class RegionDialog(QDialog):
    def __init__(self, image, title, parent):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 650)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('拖动框住目标区域；尽量贴近牌面，避开昵称、按钮和其他手牌。'))
        self.view = RegionView(image)
        layout.addWidget(self.view)
        button = QPushButton('使用这个区域')
        button.clicked.connect(self.accept)
        layout.addWidget(button)

    def normalized(self):
        r = self.view.selection
        if r is None or r.width() < 8 or r.height() < 8:
            raise ValueError('请先拖动框选一个有效区域')
        return [r.x() / self.view.image.width, r.y() / self.view.image.height,
                r.width() / self.view.image.width, r.height() / self.view.image.height]


class Signals(QObject):
    done = Signal(object)


class Task(QRunnable):
    def __init__(self, kind, fn, generation):
        super().__init__()
        self.kind, self.fn, self.generation = kind, fn, generation
        self.cancelled = False
        self.signals = Signals()

    def run(self):
        if self.cancelled:
            return
        try:
            result = {'result': self.fn()}
        except Exception as error:
            result = {'error': str(error)}
        if not self.cancelled:
            self.signals.done.emit({'task': self, 'kind': self.kind, 'generation': self.generation, **result})


class Sidecar(QMainWindow):
    model_phase = Signal(str)

    def __init__(self, demo=False):
        super().__init__()
        self.setWindowTitle('Jev 雀魂助手')
        self.resize(360, 350)
        self.setMinimumWidth(320)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.adviser = Adviser()
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.model_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='jev-model')
        self.jobs, self.busy = set(), set()
        self.generation = 0
        self.image = None
        self.image_source = 'screen'
        self.river_known = set()
        self.regions, self.capture_rect = {}, None
        self.last_signature, self.stable = None, 0
        self.last_analyzed = None
        self.last_capture = None
        self.last_update = None
        self.frame_started = None
        self.last_latency = None
        self.screen_reader = ScreenStateReader(CONFIG.parent / 'screen-state.json')
        self.reaction = None
        self.auto_melds = None
        self.forbidden_discards = []
        self.screen_riichi = {}
        self.autoplayer = AutoPlayer()
        self.riichi_flow = RiichiFlow()
        self.auto_result = None
        self.auto_hand = None
        self.auto_screen = None
        self.latest_detections = []
        self.latest_screen = {}
        self.auto_ready_after = 0.
        self.strategy_hand_id = None
        self.current_decision = None
        self.pending_vision = None
        self.window_capture = GameWindowCapture()
        self.window_table_rect = None
        self.recorder = DecisionRecorder(os.environ['JEV_DATASET_DIR']) if os.environ.get('JEV_DATASET_DIR') else None
        self.current_sample = None
        self.build()
        self.model_phase.connect(self.readiness.setText, Qt.ConnectionType.QueuedConnection)
        self.load_settings()
        self.regions.setdefault('hand', {'rect': [.10, .76, .76, .23], 'rotation': 0, 'include_drawn': True})
        self.templates = Templates(CONFIG.parent / 'templates')
        self.detector = TileDetector(self.detector_path.text(), self.templates)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self.live.setChecked(True)
        if demo:
            self.demo()
        self.exclude_capture()

    def build(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        title = QLabel('Jev  雀魂助手')
        title.setStyleSheet('font-size:17px;font-weight:700;')
        header.addWidget(title)
        header.addStretch()
        settings_button = QPushButton('设置')
        settings_button.clicked.connect(self.open_settings)
        header.addWidget(settings_button)
        layout.addLayout(header)
        self.auto_execute = QCheckBox('自动执行推荐 · Esc 停止')
        self.auto_execute.setToolTip('仅在雀魂前台执行；切到其他应用暂停点击，后台截图识别继续。')
        self.auto_execute.toggled.connect(lambda enabled: (self.autoplayer.reset(), self.riichi_flow.reset()))
        layout.addWidget(self.auto_execute)
        self.readiness = QLabel('Jev 未就绪 · 启动后自动加载')
        self.readiness.setWordWrap(True)
        self.readiness.setObjectName('muted')
        layout.addWidget(self.readiness)
        self.stop_timer = QTimer(self)
        self.stop_timer.setInterval(100)
        self.stop_timer.timeout.connect(self.check_auto_stop)
        self.stop_timer.start()
        self.status = QLabel('自动连接雀魂窗口…')
        self.status.setWordWrap(True)
        self.status.setObjectName('muted')
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle('Jev · 设置与校准')
        self.settings_dialog.resize(660, 740)
        settings_layout = QVBoxLayout(self.settings_dialog)
        self.tabs = QTabWidget()
        settings_layout.addWidget(self.tabs)
        recommend, r = self.tab('截图与校准')
        automatic_hint = QLabel('自动识别雀魂窗口并后台截图，无需框选桌面。窗口布局变化时可在这里调整区域。')
        automatic_hint.setWordWrap(True)
        r.addWidget(automatic_hint)
        toolbar = QHBoxLayout()
        for label, callback in [('框选牌桌', self.select_table), ('打开截图', self.open_image), ('截取一次', self.capture_once)]:
            b = QPushButton(label)
            b.clicked.connect(callback)
            toolbar.addWidget(b)
        r.addLayout(toolbar)
        self.live = QCheckBox('自动运行 · 每秒截图 → 识别手牌 → Jev 推荐')
        self.live.toggled.connect(self.live_changed)
        r.addWidget(self.live)
        self.activity = QLabel('尚未截图 · 尚无推荐')
        self.activity.setWordWrap(True)
        self.activity.setObjectName('muted')
        self.preview = QLabel('截图预览')
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(130)
        self.preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.preview.setStyleSheet('background:#0b111a;border-radius:8px;')
        r.addWidget(self.preview)
        zone_row = QHBoxLayout()
        self.zone = QComboBox()
        for key, value in ZONE_NAMES.items():
            self.zone.addItem(value, key)
        zone_row.addWidget(self.zone)
        b = QPushButton('框选区域')
        b.clicked.connect(self.select_zone)
        zone_row.addWidget(b)
        r.addLayout(zone_row)
        self.hand = QLineEdit()
        self.hand.setPlaceholderText('例如 123456m345p67s115z；也可识别后点牌修正')
        self.hand.textEdited.connect(self.invalidate)
        self.hand.textEdited.connect(lambda text: self.lock.setChecked(True))
        r.addWidget(self.hand)
        self.tile_grid = QGridLayout()
        r.addLayout(self.tile_grid)
        action_row = QHBoxLayout()
        self.lock = QCheckBox('手动输入模式（暂停自动更新）')
        self.lock.toggled.connect(self.manual_changed)
        action_row.addWidget(self.lock)
        analyze = QPushButton('分析这手牌')
        analyze.setObjectName('primary')
        analyze.clicked.connect(self.analyze)
        action_row.addWidget(analyze)
        r.addLayout(action_row)
        teach = QPushButton('把已校正手牌保存为当前牌面模板')
        teach.clicked.connect(self.teach_templates)
        r.addWidget(teach)
        card = QFrame()
        card.setObjectName('card')
        c = QVBoxLayout(card)
        self.engine_label = QLabel('Jev：准备加载到显卡')
        c.addWidget(self.engine_label)
        self.hero = QLabel('等待手牌')
        self.hero.setWordWrap(True)
        self.hero.setObjectName('hero')
        c.addWidget(self.hero)
        self.overall_strategy = QLabel('整体策略：等待手牌')
        self.overall_strategy.setWordWrap(True)
        c.addWidget(self.overall_strategy)
        self.explanation = QLabel('框选完成后自动运行，无需点击“分析这手牌”。')
        self.explanation.setWordWrap(True)
        c.addWidget(self.explanation)
        layout.addWidget(card)
        self.hero.setStyleSheet('font-size:32px;font-weight:700;')
        self.results = QTableWidget(0, 4)
        self.results.setHorizontalHeaderLabels(['候选', '向听', '进张', '偏好分'])
        self.results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results.verticalHeader().hide()
        self.results.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results.setMaximumHeight(145)
        layout.addWidget(self.results)
        layout.addWidget(self.status)
        layout.addWidget(self.activity)
        layout.addStretch()
        self.notes = QLabel('模型偏好分不代表和牌率或胜率。')
        self.notes.setWordWrap(True)
        self.notes.setObjectName('muted')
        r.addWidget(self.notes)
        r.addStretch()
        _, context = self.tab('牌桌信息')
        context.addWidget(QLabel('牌河按自己、下家、对家、上家排列。留空表示未知。'))
        self.rivers, self.reach = [], []
        for label in ['自己', '下家', '对家', '上家']:
            row = QHBoxLayout()
            edit, reach = QLineEdit(), QCheckBox(label + '立直')
            edit.setPlaceholderText(label + '牌河：1m 9p 东 …')
            edit.textEdited.connect(self.invalidate)
            reach.toggled.connect(self.invalidate)
            row.addWidget(edit)
            row.addWidget(reach)
            context.addLayout(row)
            self.rivers.append(edit)
            self.reach.append(reach)
        self.dora = QLineEdit()
        self.dora.setPlaceholderText('宝牌指示牌，例如4m（不是实际宝牌5m）')
        self.dora.textEdited.connect(self.invalidate)
        context.addWidget(self.dora)
        self.scores = QLineEdit('25000 25000 25000 25000')
        self.scores.textEdited.connect(self.invalidate)
        context.addWidget(QLabel('分数：自己 / 下家 / 对家 / 上家'))
        context.addWidget(self.scores)
        form = QFormLayout()
        self.dealer = QComboBox()
        self.dealer.addItems(['自己', '下家', '对家', '上家'])
        self.wind = QComboBox()
        self.wind.addItems(['东风', '南风', '西风', '北风'])
        self.honba, self.sticks = QSpinBox(), QSpinBox()
        self.hand_number = QSpinBox()
        self.hand_number.setRange(1, 4)
        self.hand_number.valueChanged.connect(self.invalidate)
        for widget in [self.honba, self.sticks]:
            widget.setRange(0, 99)
            widget.valueChanged.connect(self.invalidate)
        self.dealer.currentIndexChanged.connect(self.invalidate)
        self.wind.currentIndexChanged.connect(self.invalidate)
        for label, widget in [('庄家', self.dealer), ('场风', self.wind), ('第几局', self.hand_number), ('本场', self.honba), ('立直棒', self.sticks)]:
            form.addRow(label, widget)
        context.addLayout(form)
        context.addWidget(QLabel('本场和立直棒随截图自动更新。'))
        context.addWidget(QLabel('副露：每组用逗号分隔，例如123m,555p。暗杠前加a:。'))
        self.meld_edits = []
        for who in ['自己', '下家', '对家', '上家']:
            edit = QLineEdit()
            edit.setPlaceholderText(who + '副露，例如123m,555p')
            edit.textEdited.connect(self.invalidate)
            context.addWidget(edit)
            self.meld_edits.append(edit)
        self.confirmed = QCheckBox('已核对牌河、分数、立直、副露与宝牌')
        self.confirmed.toggled.connect(self.invalidate)
        context.addWidget(self.confirmed)
        context.addStretch()
        _, settings = self.tab('设置')
        self.monitor = QComboBox()
        self.monitor.addItems([screen.name() for screen in QApplication.screens()])
        self.monitor.hide()
        settings.addWidget(QLabel('截图来源：自动连接雀魂窗口，切换到其他应用仍继续捕获'))
        self.mode = QComboBox()
        self.mode.addItems(['牌理引擎', 'Jev Play v1（GPU）'])
        self.mode.setCurrentIndex(1)
        self.mode.currentIndexChanged.connect(self.invalidate)
        settings.addWidget(QLabel('推荐引擎'))
        settings.addWidget(self.mode)
        checkpoint = ROOT / 'models/jev-discard-v1'
        self.checkpoint = QLineEdit(str(checkpoint))
        settings.addWidget(QLabel('本地Jev检查点目录'))
        settings.addWidget(self.checkpoint)
        self.load_button = QPushButton('加载 Jev 到 NVIDIA GPU')
        self.load_button.clicked.connect(self.load_model)
        settings.addWidget(self.load_button)
        path = ROOT / 'assets/tiles.onnx'
        if not path.exists():
            path = ROOT / 'runs/windows-assets/tiles.onnx'
        self.detector_path = QLineEdit(str(path))
        settings.addWidget(QLabel('麻将牌识别模型'))
        settings.addWidget(self.detector_path)
        settings.addWidget(QLabel('首次加载Jev会下载固定版本的Qwen底座；之后可使用本地缓存。'))
        for label, callback in [('保存校准设置', self.save_settings), ('靠右停靠', self.dock), ('体验演示手牌', self.demo)]:
            b = QPushButton(label)
            b.clicked.connect(callback)
            settings.addWidget(b)
        settings.addStretch()
        self.setCentralWidget(root)

    def open_settings(self):
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def closeEvent(self, event):
        self.stop_timer.stop()
        if hasattr(self, 'timer'):
            self.timer.stop()
        self.pending_vision = None
        for task in self.jobs:
            task.cancelled = True
        self.window_capture.stop()
        self.model_pool.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)

    def tab(self, title):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        box = QVBoxLayout(body)
        scroll.setWidget(body)
        self.tabs.addTab(scroll, title)
        return body, box

    def invalidate(self, *args):
        self.generation += 1
        self.last_analyzed = None
        self.auto_result = None
        if 'load' in self.busy:
            text = 'Jev 加载中'
        elif self.lock.isChecked():
            text = '手动输入模式'
        elif not self.live.isChecked():
            text = '自动更新已暂停'
        elif self.capture_rect is None or 'hand' not in self.regions:
            text = '请框选手牌区域'
        else:
            text = '自动识别中'
        self.hero.setText(text)
        self.results.setRowCount(0)

    def live_changed(self, enabled):
        if enabled:
            self.lock.setChecked(False)
        self.invalidate()
        if enabled and hasattr(self, 'timer'):
            QTimer.singleShot(0, self.tick)
        elif not enabled:
            self.pending_vision = None
            self.status.setText('自动更新已暂停；勾选“自动运行”即可恢复')

    def manual_changed(self, enabled):
        if enabled:
            self.live.setChecked(False)
        self.invalidate()

    def update_activity(self):
        timing = f' · 截图到建议 {self.last_latency:.2f}s' if self.last_latency is not None else ''
        self.activity.setText(f'最近截图：{self.last_capture or "尚未截图"} · 最近推荐：{self.last_update or "尚未生成"}{timing}')

    def wait_for_frame(self, title, detail):
        if self.hero.text() != title:
            self.invalidate()
        self.hero.setText(title)
        if not self.current_decision:
            self.explanation.setText(detail)
        self.status.setText(detail)

    def debug_event(self, event_kind, **values):
        directory = os.environ.get('JEV_DEBUG_DIR')
        if directory:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            with (path / 'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'time': datetime.now().isoformat(), 'event': event_kind, **values}, ensure_ascii=False) + '\n')

    def start(self, kind, fn):
        if kind in self.busy or (kind in ['analysis', 'load'] and self.busy & {'analysis', 'load'}):
            return
        task = Task(kind, fn, self.generation)
        task.frame_started = self.frame_started
        task.sample_id = self.current_sample
        self.jobs.add(task)
        self.busy.add(kind)
        task.signals.done.connect(self.finished, Qt.ConnectionType.QueuedConnection)
        if kind in ('load', 'analysis'):
            self.model_pool.submit(task.run)
        else:
            self.pool.start(task)

    @Slot(object)
    def finished(self, message):
        kind = message['kind']
        self.jobs.discard(message['task'])
        self.busy.discard(kind)
        if kind == 'vision':
            QTimer.singleShot(0, self.start_pending_vision)
        if 'error' in message:
            self.debug_event('error', kind=kind, error=message['error'])
            if kind in ['vision', 'analysis'] and message['generation'] != self.generation:
                return
            self.status.setText(message['error'])
            if kind == 'analysis':
                self.last_analyzed = None
                self.analysis_error(message['error'])
            if kind == 'vision':
                self.wait_for_frame('识别重试中', '本帧未完成识别，下一秒自动重试：' + message['error'])
            if kind == 'load':
                self.readiness.setText('Jev 未就绪 · 加载或预热失败，请重试')
                self.load_button.setEnabled(True)
                self.engine_label.setText('Jev：加载失败，可在设置中重试')
                self.hero.setText('Jev 加载失败')
                self.explanation.setText('请到设置重试加载 Jev；截图识别仍会每秒继续。')
            return
        if kind == 'vision' and message['generation'] == self.generation:
            self.frame_started = message['task'].frame_started
            self.recognized(message['result'])
        elif kind == 'load':
            self.load_button.setEnabled(True)
            self.mode.setCurrentIndex(1)
            self.engine_label.setText(message['result'])
            self.readiness.setText('Jev 已就绪 · 可以开始匹配')
            self.status.setText('模型已加载并完成实际推理预热')
            self.debug_event('model_ready', description=message['result'])
            self.last_analyzed = None
            if self.live.isChecked():
                QTimer.singleShot(0, self.tick)
        elif kind == 'analysis' and message['generation'] == self.generation:
            started = getattr(message['task'], 'frame_started', None)
            self.last_latency = time.monotonic() - started if started is not None else None
            self.show_result(message['result'])
            if self.recorder and message['task'].sample_id:
                self.recorder.prediction(message['task'].sample_id, message['result'], self.last_latency)

    def screen_image(self):
        return self.window_capture.snapshot()

    def analysis_error(self, error):
        if '赤五' in error or '同一种牌' in error or '牌代码' in error:
            title = '手牌解析中'
            detail = '手牌牌面解析未完成：' + error
        elif '公开牌面' in error:
            title = '公开牌面读取中'
            detail = '手牌已读取，牌河或宝牌需要重新读取'
        elif '副露' in error:
            title = '副露读取中'
            detail = '手牌已读取，正在补全副露信息'
        elif '加载' in error or 'CUDA' in error:
            title = 'Jev 未就绪'
            detail = error
        else:
            title = '分析未完成'
            detail = error
        self.hero.setText(title)
        self.status.setText(detail + (' · 自动读取下一帧' if self.live.isChecked() else ''))
        self.explanation.setText('')

    def select_table(self):
        try:
            image = self.screen_image()
            dialog = RegionDialog(image, '框选整个雀魂牌桌', self)
            if dialog.exec():
                self.capture_rect = dialog.normalized()
                self.window_table_rect = self.capture_rect
                self.image_source = 'screen'
                self.regions = {}
                self.last_signature, self.stable = None, 0
                self.river_known.clear()
                self.screen_reader = ScreenStateReader()
                self.image = self.crop_screen(image)
                self.update_preview()
                self.invalidate()
                self.status.setText('牌桌已选好；请选择“自己的手牌”并框选区域')
        except Exception as error:
            self.status.setText(str(error))

    def crop_screen(self, image):
        if self.window_table_rect is None:
            return image
        x, y, w, h = self.window_table_rect
        return image.crop((int(x * image.width), int(y * image.height), int((x + w) * image.width), int((y + h) * image.height)))

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, '打开雀魂截图', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if path:
            self.live.setChecked(False)
            self.image = Image.open(path).convert('RGB')
            self.image_source = 'file'
            self.invalidate()
            self.update_preview()
            self.status.setText('截图已打开；校准手牌区域后点“截取一次”识别')

    def select_zone(self):
        try:
            if self.image is None:
                raise ValueError('请先截取牌桌或打开一张截图')
            key = self.zone.currentData()
            dialog = RegionDialog(self.image, '框选：' + ZONE_NAMES[key], self)
            if dialog.exec():
                rotation = 0
                if key in ['river1', 'river2', 'river3']:
                    choice, ok = QInputDialog.getItem(self, '把牌面转正', '区域内的牌需要逆时针旋转多少度？', ['0', '90', '180', '270'], 0, False)
                    if ok:
                        rotation = int(choice)
                self.regions[key] = {'rect': dialog.normalized(), 'rotation': rotation}
                if key == 'hand':
                    self.regions[key] = include_drawn_tile(self.regions[key])
                self.last_signature, self.stable = None, 0
                self.save_settings()
                self.invalidate()
                if self.image_source == 'screen' and self.capture_rect is not None and 'hand' in self.regions:
                    self.lock.setChecked(False)
                    self.live.setChecked(True)
                    self.status.setText('区域已保存，已自动开始每秒截图、识别和 Jev 推荐')
                    QTimer.singleShot(0, self.tick)
                elif self.image_source == 'file':
                    self.capture_once()
        except Exception as error:
            self.status.setText(str(error))

    def update_preview(self):
        if self.image is not None:
            self.preview.setPixmap(pixmap(self.image).scaled(self.preview.width(), 165, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'preview'):
            QTimer.singleShot(0, self.update_preview)

    def capture_once(self):
        self.capture(manual=True)

    def capture(self, manual=False):
        try:
            self.frame_started = time.monotonic()
            if self.live.isChecked() or self.image is None or self.image_source != 'file':
                self.image = self.crop_screen(self.screen_image())
            if 'hand' not in self.regions:
                raise ValueError('请先框选自己的手牌区域')
            self.update_preview()
            self.last_capture = datetime.now().strftime('%H:%M:%S')
            self.update_activity()
            if os.environ.get('JEV_DEBUG_DIR'):
                debug_path = Path(os.environ['JEV_DEBUG_DIR'])
                debug_path.mkdir(parents=True, exist_ok=True)
                if self.recorder:
                    preview = self.image.copy()
                    preview.thumbnail((1280, 1280))
                    preview.save(debug_path / 'latest-table-next.jpg', quality=68, optimize=True)
                    (debug_path / 'latest-table-next.jpg').replace(debug_path / 'latest-table.jpg')
                else:
                    self.image.save(debug_path / 'latest-table-next.png')
                    (debug_path / 'latest-table-next.png').replace(debug_path / 'latest-table.png')
            if manual or (self.results.rowCount() == 0 and not self.busy & {'analysis', 'load'}):
                self.status.setText('每秒监测牌局中…')
            image, regions = self.image.copy(), copy.deepcopy(self.regions)
            for key, region in DEFAULT_RIVERS.items():
                regions.setdefault(key, region)
            self.detector.path = Path(self.detector_path.text())
            self.pending_vision = (image, regions, manual, self.frame_started)
            self.start_pending_vision()
        except Exception as error:
            title = '等待雀魂窗口' if isinstance(error, CaptureUnavailable) else '截图未完成'
            self.wait_for_frame(title, str(error))

    def start_pending_vision(self):
        if 'vision' in self.busy or self.pending_vision is None:
            return
        image, regions, manual, started = self.pending_vision
        self.pending_vision = None
        self.frame_started = started
        self.start('vision', lambda: self.recognize_frame(image, regions, manual))

    def tick(self):
        if self.live.isChecked():
            self.capture()

    def recognize_frame(self, image, regions, manual):
        found = self.detector.regions(image, {k: v for k, v in regions.items() if k not in ('meld0','counters')})
        found['_screen'] = self.screen_reader.read(image, regions, self.detector, found)
        found['_image_size'] = image.size
        if self.recorder and (len(found.get('hand', [])) in (2, 5, 8, 11, 14) or found['_screen']['offered']):
            labels = {'hand': [d.tile for d in found.get('hand', [])], 'screen': found['_screen'],
                      'rivers': {k: [d.tile for d in v] for k, v in found.items() if k.startswith('river')}}
            found['_sample'] = self.recorder.record(image, labels)
        if not self.recorder and os.environ.get('JEV_DEBUG_DIR') and (len(found.get('hand', [])) in (2, 5, 8, 11, 14) or found['_screen']['offered']):
            signature = repr(([d.tile for d in found.get('hand', [])], found['_screen']))
            key = hashlib.sha256(signature.encode()).hexdigest()[:16]
            path = Path(os.environ['JEV_DEBUG_DIR']) / ('decision-' + key + '.png')
            if not path.exists():
                image.save(path)
        return found, manual

    def recognized(self, result):
        regions, manual = result
        self.current_sample = regions.get('_sample')
        screen = regions.get('_screen', {})
        self.update_counters(screen.get('counters', {}))
        hand_id = screen.get('hand_id')
        if hand_id is not None and hand_id != self.strategy_hand_id:
            if self.strategy_hand_id is not None:
                self.overall_strategy.setText('整体策略：新一局，正在计算')
                self.current_decision = None
                self.explanation.setText('当前决策：等待新一局推荐')
            self.strategy_hand_id = hand_id
        self.latest_screen = screen
        self.latest_image_size = regions.get('_image_size', self.image.size if self.image is not None else (1, 1))
        reaction = screen.get('reaction')
        if reaction != self.reaction:
            self.reaction = reaction
            self.invalidate()
        self.auto_melds = screen.get('melds')
        riichi_state = {key: bool(screen.get(key)) for key in ('riichi_available', 'riichi_selecting', 'own_riichi')}
        if riichi_state != self.screen_riichi:
            self.screen_riichi = riichi_state
            self.invalidate()
        self.forbidden_discards = screen.get('forbidden', [])
        detections = regions.get('hand', [])
        self.latest_detections = detections
        signature = tuple(d.tile for d in detections)
        click_progress = self.autoplayer.observe(signature, repr(screen))
        if click_progress:
            self.status.setText(click_progress)
            if self.autoplayer.pending is None:
                self.debug_event('auto_click_state_changed', hand=list(signature))
        self.debug_event('recognized', hand=list(signature), screen=screen)
        self.stable = self.stable + 1 if signature == self.last_signature else 1
        self.last_signature = signature
        if not self.lock.isChecked():
            text = ' '.join(signature)
            if self.hand.text() != text:
                self.hand.setText(text)
                self.invalidate()
                self.paint_tiles(list(signature))
        for key, items in regions.items():
            if key.startswith('river'):
                self.river_known.add(int(key[-1]))
                target = self.rivers[int(key[-1])]
            elif key == 'dora':
                target = self.dora
            else:
                continue
            text = ' '.join(d.tile for d in items)
            if not self.live.isChecked() and not items and target.text():
                continue  # A transient missed detection is not proof that a river was cleared.
            if target.text() != text:
                target.setText(text)
                if not self.live.isChecked():
                    self.invalidate()
        if self.lock.isChecked():
            self.status.setText('手动输入模式；勾选“自动运行”恢复每秒自动更新')
            return
        if self.try_riichi_selection():
            return
        if not detections:
            self.wait_for_frame('正在寻找手牌', '尚未读到手牌，下一秒自动截图重试，无需操作')
            return
        if len(detections) in (1, 4, 7, 10, 13):
            if screen.get('offered'):
                if self.reaction:
                    self.analyze(automatic=not manual)
                    self.try_auto_execute()
                else:
                    self.wait_for_frame('正在读取吃碰牌面', '已检测到动作按钮，正在自动读取对应牌面')
                return
            self.wait_for_frame('等待摸牌', f'当前{len(detections)}张手牌；摸牌后自动用 Jev 推荐，无需点击')
            return
        if len(detections) not in (2, 5, 8, 11, 14):
            self.wait_for_frame('等待完整手牌', f'本帧读到{len(detections)}张，下一秒自动重新识别')
            return
        if 'load' in self.busy:
            self.hero.setText('Jev 加载中')
            self.status.setText('手牌已识别；Jev 加载完成后自动推荐，无需点击')
            return
        self.analyze(automatic=not manual)
        self.try_auto_execute()

    def update_counters(self, values):
        changed = False
        for key, widget in (('honba', self.honba), ('sticks', self.sticks)):
            value = values.get(key)
            if value is not None and widget.value() != value:
                blocked = widget.blockSignals(True)
                widget.setValue(value)
                widget.blockSignals(blocked)
                changed = True
        if changed:
            self.invalidate()

    def paint_tiles(self, tiles):
        while self.tile_grid.count():
            widget = self.tile_grid.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        for i, tile in enumerate(tiles):
            button = QPushButton(name(tile))
            button.setToolTip(tile)
            button.setStyleSheet('padding:7px 2px;')
            button.clicked.connect(lambda checked=False, position=i: self.edit_tile(position))
            self.tile_grid.addWidget(button, i // 7, i % 7)

    def edit_tile(self, position):
        try:
            tiles = parse(self.hand.text())
            text, ok = QInputDialog.getText(self, '校正这张牌', '牌代码，例如1m、0p、东、白：', text=tiles[position])
            if ok:
                replacement = parse(text)
                if len(replacement) != 1:
                    raise ValueError('一次只校正一张牌')
                tiles[position] = replacement[0]
                parse(' '.join(tiles))
                self.lock.setChecked(True)
                self.hand.setText(' '.join(tiles))
                self.paint_tiles(tiles)
                self.invalidate()
        except Exception as error:
            self.status.setText(str(error))

    def teach_templates(self):
        try:
            if self.image is None or 'hand' not in self.regions:
                raise ValueError('请先打开截图或截取牌桌，并校准手牌区域')
            x, y, w, h = self.regions['hand']['rect']
            crop = self.image.crop((round(x * self.image.width), round(y * self.image.height),
                                    round((x + w) * self.image.width), round((y + h) * self.image.height)))
            tiles = parse(self.hand.text())
            self.templates.teach(crop, hand_boxes(crop), tiles)
            self.status.setText(f'已保存{len(tiles)}张校正模板；取消手牌锁定后继续实时识别')
        except Exception as error:
            self.status.setText(str(error))

    def board(self):
        hand = parse(self.hand.text())
        melds = [parse_melds(edit.text()) for edit in self.meld_edits]
        if self.live.isChecked() and self.auto_melds is not None:
            melds[0] = self.auto_melds
        return Board(hand=hand, rivers=[parse(e.text()) if e.text().strip() or i in self.river_known or self.confirmed.isChecked() else None
                                      for i, e in enumerate(self.rivers)],
                     indicators=parse(self.dora.text()), melds=melds,
                     riichi=sorted(set([i for i, c in enumerate(self.reach) if c.isChecked()] +
                                       ([0] if self.live.isChecked() and self.screen_riichi.get('own_riichi') else []))),
                     scores=[int(x) for x in self.scores.text().split()], dealer=self.dealer.currentIndex(),
                     round_wind=self.wind.currentIndex(), honba=self.honba.value(), sticks=self.sticks.value(),
                     drawn=hand[-1] if hand else None, metadata_confirmed=self.confirmed.isChecked(), hand_number=self.hand_number.value(),
                     reaction=self.reaction if self.live.isChecked() else None,
                     forbidden_discards=self.forbidden_discards if self.live.isChecked() else [],
                     riichi_available=self.live.isChecked() and self.screen_riichi.get('riichi_available', False),
                     riichi_selecting=self.live.isChecked() and self.screen_riichi.get('riichi_selecting', False))

    def analyze(self, checked=False, automatic=False):
        if self.busy & {'analysis', 'load'}:
            return
        try:
            board = self.board()
            try:
                board.visible()
            except ValueError:
                # An inconsistent public crop must not discard a valid hand.
                board.rivers = [None] * 4
                board.indicators = []
                board.melds[1:] = [[], [], []]
                board.metadata_confirmed = False
                board.visible()
            # Public river OCR can jitter while nothing about our legal decision
            # changes. Calculate once per decision, not once per screenshot.
            signature = (repr((board.hand, board.reaction, board.melds[0], board.riichi,
                               board.riichi_available, board.riichi_selecting, board.forbidden_discards,
                               board.honba, board.sticks))
                         if automatic else repr(board)) + str(self.mode.currentIndex())
            if signature == self.last_analyzed:
                return
            same_advice = signature == self.last_analyzed and self.results.rowCount() > 0
            self.last_analyzed = signature
            use_model = self.mode.currentIndex() == 1
            if not same_advice:
                self.hero.setText('Jev 运算中' if use_model else '牌理分析中')
            self.status.setText('后台更新中 · 当前建议仍适用' if same_advice else
                                '手牌已识别，正在用 Jev 计算推荐…' if use_model else '手牌已识别，正在分析…')
            self.start('analysis', lambda: self.adviser.recommend(board, use_model))
        except Exception as error:
            if automatic:
                self.analysis_error(str(error))
            else:
                self.status.setText(str(error))

    def show_result(self, result):
        self.auto_result = result
        self.auto_hand = tuple(d.tile for d in self.latest_detections)
        self.auto_screen = repr(self.latest_screen)
        self.auto_ready_after = time.monotonic()
        top = result['top3'][0]
        self.hero.setText(('建议：' if result.get('reaction') else '建议打 ') + top['name'])
        if top.get('action', {}).get('type') == 'reach':
            self.hero.setText(top['name'])
        elif result.get('riichi_selecting'):
            self.hero.setText('立直选牌 → 打 ' + top['name'])
        device = ' · ' + self.adviser.device.upper() if self.adviser.device and result['engine'].startswith('Jev') else ''
        self.engine_label.setText(result['engine'] + device + ' · ' + datetime.now().strftime('%H:%M:%S'))
        safe = '，对立直家现物' if set(top['safe']) & {i for i, c in enumerate(self.reach) if c.isChecked()} else ''
        self.explanation.setText('Jev 已比较当前吃碰与跳过选项；操作后自动识别副露并继续推荐弃牌。' if result.get('reaction')
                                 else f'弃牌后{top["sh"]}向听，{top["ukeire"]}枚可见剩余进张{safe}。')
        if top.get('action', {}).get('type') == 'reach':
            self.explanation.setText('先点击游戏的“立直”，再单击推荐牌；选牌画面自动继续识别。')
        elif result.get('riichi_selecting'):
            self.explanation.setText('已进入立直选牌，只推荐能听牌的弃牌；单击推荐牌完成立直。')
        if result.get('direction'):
            text = '整体策略：' + result['direction']
            if self.overall_strategy.text() != text:
                self.overall_strategy.setText(text)
        if result.get('decision'):
            self.current_decision = result['decision']
            self.explanation.setText('当前决策：' + self.current_decision)
        self.results.setRowCount(len(result['top3']))
        for row, candidate in enumerate(result['top3']):
            items = [candidate['name'], str(candidate['sh']) if candidate['sh'] is not None else '—',
                     str(candidate['ukeire']) if candidate['ukeire'] is not None else '—',
                     f'{candidate["preference"]:.1%}' if 'preference' in candidate else '—']
            for col, value in enumerate(items):
                self.results.setItem(row, col, QTableWidgetItem(value))
        notes = result['notes'] + ['模型偏好分不是胜率。赤五为色彩估计，可点击校正。']
        if result['winning_shape']:
            self.hero.setText('和牌形 · 检查自摸')
            self.explanation.setText('若游戏提供自摸/和牌按钮，请先确认。未满足役等条件时，可参考下方弃牌候选。')
            notes.insert(0, '当前是和牌形：先检查游戏是否提供自摸/和牌按钮。')
        self.notes.setText('\n'.join(notes))
        self.last_update = datetime.now().strftime('%H:%M:%S')
        self.debug_event('recommendation', hand=self.hand.text(), engine=result['engine'], latency=self.last_latency,
                         top3=result['top3'], strategy=result.get('strategy'),
                         direction=result.get('direction'), decision=result.get('decision'))
        self.update_activity()
        self.status.setText('建议已更新 · 监测出牌完成，下一次决策自动刷新' if self.live.isChecked() else '建议已更新 · 手动模式')

    def check_auto_stop(self):
        if self.auto_execute.isChecked() and sys.platform == 'win32' and escape_pressed():
            self.auto_execute.setChecked(False)
            self.status.setText('Esc 已停止自动点击；截图识别继续')

    def try_auto_execute(self):
        if not self.auto_execute.isChecked() or not self.live.isChecked() or not self.auto_result:
            return
        if self.frame_started is None or self.frame_started <= self.auto_ready_after:
            return  # Verify a frame captured after inference completed before input.
        hand = tuple(d.tile for d in self.latest_detections)
        if hand != self.auto_hand or repr(self.latest_screen) != self.auto_screen:
            return
        # A completed discard/call ends this decision. A later identical draw
        # receives a fresh key after invalidate clears the previous result.
        key = (self.generation, hand, self.auto_screen, repr(self.auto_result['top3'][0]))
        try:
            plan = plan_click(self.auto_result, self.latest_detections, self.latest_screen,
                              self.regions['hand']['rect'], self.latest_image_size)
            if self.autoplayer.execute(key, plan, self.window_capture, self.window_table_rect):
                top = self.auto_result['top3'][0]
                if top.get('action', {}).get('type') == 'reach':
                    self.riichi_flow.begin(hand, top['tile'])
                    self.screen_reader.expected_riichi_hand = list(hand)
                    self.screen_reader.expected_riichi_tile = top['tile']
                self.status.setText('已自动点击：' + plan.label + ' · 等待操作完成')
                self.debug_event('auto_click', hand=list(hand), action=plan.label, x=plan.x, y=plan.y)
            elif plan is None:
                self.status.setText('该动作需要选择吃碰组合，自动点击等待')
        except RuntimeError as error:
            self.status.setText(str(error))

    def try_riichi_selection(self):
        flow = self.riichi_flow
        if not self.auto_execute.isChecked() or flow.hand is None:
            return False
        plan = flow.plan(self.latest_detections, self.latest_screen,
                         self.regions['hand']['rect'], self.latest_image_size)
        if flow.hand is None:
            return False
        self.hero.setText('立直第2步 → 打 ' + name(flow.tile))
        self.status.setText('等待弃牌完成' if flow.clicked else '等待立直选牌界面，保留 Jev 已选弃牌')
        if plan is not None and self.frame_started is not None and self.frame_started > flow.started:
            key = ('riichi-discard', tuple(d.tile for d in self.latest_detections), repr(self.latest_screen), flow.started)
            try:
                if self.autoplayer.execute(key, plan, self.window_capture, self.window_table_rect):
                    flow.clicked = True
                    self.status.setText('已确认立直弃牌：' + name(flow.tile) + ' · 等待牌离手')
                    self.debug_event('riichi_discard_click', tile=flow.tile, x=plan.x, y=plan.y)
            except RuntimeError as error:
                self.status.setText(str(error))
        return True

    def load_model(self):
        if self.busy & {'load', 'analysis'}:
            return
        self.readiness.setText('Jev 加载中 · 请等待就绪后再匹配')
        self.load_button.setEnabled(False)
        self.engine_label.setText('Jev：正在加载到显卡，加载完成后自动推荐')
        self.status.setText('加载Jev中；首次下载底座可能需要几分钟…')
        self.hero.setText('Jev 加载中')
        self.explanation.setText('加载完成后自动分析识别到的手牌，无需点击。')
        checkpoint = self.checkpoint.text().strip()
        self.start('load', lambda: self.adviser.prepare(checkpoint, self.model_phase.emit))

    def demo(self):
        self.live.setChecked(False)
        self.lock.setChecked(True)
        self.hand.setText(DEMO_HAND)
        self.dora.setText('4m')
        self.mode.setCurrentIndex(0)
        self.image_source = 'demo'
        self.image = Image.new('RGB', (960, 300), '#174a46')
        drawing = ImageDraw.Draw(self.image)
        drawing.text((24, 20), 'DEMO  /  local hand analysis', fill='#a8d7cb')
        drawing.rounded_rectangle((375, 55, 585, 150), radius=12, fill='#142b32')
        drawing.text((440, 92), '25000', fill='#f2d589')
        for i, tile in enumerate(parse(DEMO_HAND)):
            x = 20 + i * 65
            drawing.rounded_rectangle((x, 180, x + 56, 277), radius=5,
                                       fill='#f2efe5', outline='#67dcb5' if tile == '5z' else '#d6cdb4', width=3)
            drawing.text((x + 13, 216), tile, fill='#16483f')
        self.update_preview()
        self.invalidate()
        self.analyze()
        self.tabs.setCurrentIndex(0)

    def dock(self):
        geometry = QApplication.primaryScreen().availableGeometry()
        self.move(geometry.right() - self.width(), geometry.top() + 12)

    def save_settings(self):
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps({'regions': self.regions, 'capture_rect': self.capture_rect,
            'window_table_rect': self.window_table_rect,
            'monitor': self.monitor.currentIndex(), 'checkpoint': self.checkpoint.text(),
            'detector': self.detector_path.text()}, ensure_ascii=False, indent=2), encoding='utf-8')

    def load_settings(self):
        try:
            data = json.loads(CONFIG.read_text(encoding='utf-8'))
            self.regions, self.capture_rect = data.get('regions', {}), data.get('capture_rect')
            self.window_table_rect = data.get('window_table_rect')
            if 'hand' in self.regions:
                self.regions['hand'] = include_drawn_tile(self.regions['hand'])
            self.monitor.setCurrentIndex(min(data.get('monitor', 0), self.monitor.count() - 1))
            if Path(data.get('checkpoint', '')).is_dir():
                self.checkpoint.setText(data['checkpoint'])
            if Path(data.get('detector', '')).is_file():
                self.detector_path.setText(data['detector'])
        except (FileNotFoundError, ValueError):
            pass

    def exclude_capture(self):
        if sys.platform == 'win32':
            import ctypes
            from ctypes import wintypes
            function = ctypes.windll.user32.SetWindowDisplayAffinity
            function.argtypes = [wintypes.HWND, wintypes.DWORD]
            function.restype = wintypes.BOOL
            function(int(self.winId()), 0x11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--minimized', action='store_true', help='Keep automatic capture running in the background')
    parser.add_argument('--preview', type=Path, help='Render a demo panel to PNG, including headless environments')
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    font_path = Path(os.environ.get('JEV_FONT', ROOT / 'assets/NotoSansSC.ttf'))
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
    app.setStyleSheet(STYLE)
    window = Sidecar(demo=args.demo or bool(args.preview))
    window.dock()
    window.showMinimized() if args.minimized else window.show()
    if not args.demo and not args.preview:
        QTimer.singleShot(500, window.load_model)
    if args.preview:
        def save():
            args.preview.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.preview))
            app.quit()
        QTimer.singleShot(3000, save)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
