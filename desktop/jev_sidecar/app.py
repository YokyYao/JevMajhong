"""Always-on-top Windows side panel with calibrated local screen capture."""
import argparse
import copy
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

ROOT = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parents[2]
CONFIG = Path(os.environ.get('LOCALAPPDATA', Path.home() / '.config')) / 'JevMahjong' / 'settings.json'
DEMO_HAND = '123456m345p67s115z'
ZONE_NAMES = {'hand': '自己的手牌', 'river0': '自己的牌河', 'river1': '下家牌河', 'river2': '对家牌河',
              'river3': '上家牌河', 'dora': '宝牌指示牌'}
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
        self.signals = Signals()

    def run(self):
        try:
            result = {'result': self.fn()}
        except Exception as error:
            result = {'error': str(error)}
        self.signals.done.emit({'task': self, 'kind': self.kind, 'generation': self.generation, **result})


class Sidecar(QMainWindow):
    def __init__(self, demo=False):
        super().__init__()
        self.setWindowTitle('Jev 雀魂助手')
        self.resize(440, 940)
        self.setMinimumWidth(400)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.adviser = Adviser()
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.jobs, self.busy = set(), set()
        self.generation = 0
        self.image = None
        self.image_source = 'screen'
        self.river_known = set()
        self.regions, self.capture_rect = {}, None
        self.last_signature, self.stable = None, 0
        self.last_analyzed = None
        self.build()
        self.load_settings()
        self.templates = Templates(CONFIG.parent / 'templates')
        self.detector = TileDetector(self.detector_path.text(), self.templates)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        if demo:
            self.demo()
        self.exclude_capture()

    def build(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        title = QLabel('Jev  雀魂助手')
        title.setStyleSheet('font-size:22px;font-weight:700;')
        layout.addWidget(title)
        self.status = QLabel('先框选牌桌，再校准自己的手牌区域')
        self.status.setWordWrap(True)
        self.status.setObjectName('muted')
        layout.addWidget(self.status)
        toolbar = QHBoxLayout()
        for label, callback in [('框选牌桌', self.select_table), ('打开截图', self.open_image), ('截取一次', self.capture_once)]:
            b = QPushButton(label)
            b.clicked.connect(callback)
            toolbar.addWidget(b)
        layout.addLayout(toolbar)
        self.live = QCheckBox('实时识别 · 每秒检查一次')
        self.live.toggled.connect(self.invalidate)
        layout.addWidget(self.live)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        recommend, r = self.tab('出牌建议')
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
        self.lock = QCheckBox('锁定手牌，手动校正')
        self.lock.toggled.connect(self.invalidate)
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
        self.engine_label = QLabel('牌理引擎 · Jev可在设置中加载')
        c.addWidget(self.engine_label)
        self.hero = QLabel('等待手牌')
        self.hero.setObjectName('hero')
        c.addWidget(self.hero)
        self.explanation = QLabel('识别稳定后显示建议；点击上方牌面可以修正。')
        self.explanation.setWordWrap(True)
        c.addWidget(self.explanation)
        r.addWidget(card)
        self.results = QTableWidget(0, 4)
        self.results.setHorizontalHeaderLabels(['候选', '向听', '进张', '偏好分'])
        self.results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results.verticalHeader().hide()
        self.results.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results.setMaximumHeight(145)
        r.addWidget(self.results)
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
        settings.addWidget(QLabel('截图屏幕'))
        settings.addWidget(self.monitor)
        self.mode = QComboBox()
        self.mode.addItems(['牌理引擎', 'Jev GPU 校正版2'])
        self.mode.currentIndexChanged.connect(self.invalidate)
        settings.addWidget(QLabel('推荐引擎'))
        settings.addWidget(self.mode)
        checkpoint = ROOT / 'models/calibration-02'
        if not checkpoint.exists():
            checkpoint = ROOT / 'runs/head-rl-diagonal-value-20261030/checkpoints/calibration-02'
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
        self.hero.setText('等待更新')
        self.results.setRowCount(0)

    def start(self, kind, fn):
        if kind in self.busy or (kind in ['analysis', 'load'] and self.busy & {'analysis', 'load'}):
            return
        task = Task(kind, fn, self.generation)
        self.jobs.add(task)
        self.busy.add(kind)
        task.signals.done.connect(self.finished, Qt.ConnectionType.QueuedConnection)
        self.pool.start(task)

    @Slot(object)
    def finished(self, message):
        kind = message['kind']
        self.jobs.discard(message['task'])
        self.busy.discard(kind)
        if 'error' in message:
            if kind in ['vision', 'analysis'] and message['generation'] != self.generation:
                return
            self.status.setText(message['error'])
            if kind == 'analysis':
                self.last_analyzed = None
                self.hero.setText('请核对牌面')
            if kind == 'load':
                self.load_button.setEnabled(True)
            return
        if kind == 'vision' and message['generation'] == self.generation:
            self.recognized(message['result'])
        elif kind == 'load':
            self.load_button.setEnabled(True)
            self.mode.setCurrentIndex(1)
            self.engine_label.setText(message['result'])
            self.status.setText('Jev加载完成，可开始本机推理')
            self.last_analyzed = None
        elif kind == 'analysis' and message['generation'] == self.generation:
            self.show_result(message['result'])

    def screen_image(self):
        screens = QApplication.screens()
        screen = screens[min(self.monitor.currentIndex(), len(screens) - 1)]
        return pillow(screen.grabWindow(0).toImage())

    def select_table(self):
        try:
            image = self.screen_image()
            dialog = RegionDialog(image, '框选整个雀魂牌桌', self)
            if dialog.exec():
                self.capture_rect = dialog.normalized()
                self.image_source = 'screen'
                self.regions = {}
                self.image = self.crop_screen(image)
                self.update_preview()
                self.invalidate()
                self.status.setText('牌桌已选好；请选择“自己的手牌”并框选区域')
        except Exception as error:
            self.status.setText(str(error))

    def crop_screen(self, image):
        if self.capture_rect is None:
            raise ValueError('请先框选牌桌')
        x, y, w, h = self.capture_rect
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
                self.save_settings()
                self.invalidate()
                self.status.setText(ZONE_NAMES[key] + '区域已保存')
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
        if 'vision' in self.busy:
            return
        try:
            if self.live.isChecked() or self.image is None or self.image_source != 'file':
                self.image = self.crop_screen(self.screen_image())
            if 'hand' not in self.regions:
                raise ValueError('请先框选自己的手牌区域')
            self.update_preview()
            image, regions = self.image.copy(), copy.deepcopy(self.regions)
            self.detector.path = Path(self.detector_path.text())
            self.start('vision', lambda: (self.detector.regions(image, regions), manual))
        except Exception as error:
            self.status.setText(str(error))

    def tick(self):
        if self.live.isChecked():
            self.capture()

    def recognized(self, result):
        regions, manual = result
        detections = regions.get('hand', [])
        signature = tuple(d.tile for d in detections)
        self.stable = self.stable + 1 if signature == self.last_signature else 1
        self.last_signature = signature
        if not self.lock.isChecked():
            text = ' '.join(signature)
            if self.hand.text() != text:
                self.hand.setText(text)
                self.invalidate()
            self.paint_tiles(list(signature), [d.confidence for d in detections])
        for key, items in regions.items():
            if key.startswith('river'):
                self.river_known.add(int(key[-1]))
                target = self.rivers[int(key[-1])]
            elif key == 'dora':
                target = self.dora
            else:
                continue
            if items and min(d.confidence for d in items) < .70:
                continue
            text = ' '.join(d.tile for d in items)
            if not items and target.text():
                continue  # A transient missed detection is not proof that a river was cleared.
            if target.text() != text:
                target.setText(text)
                self.invalidate()
        if not detections or min(d.confidence for d in detections) < .70:
            self.hero.setText('请校正识别')
            self.status.setText('存在低置信度牌面；点击牌面修正，再按“分析这手牌”')
            return
        self.status.setText(f'手牌识别{len(detections)}张 · 连续稳定{self.stable}帧')
        if self.lock.isChecked():
            self.status.setText('手牌已锁定：可手动分析；取消锁定后恢复自动更新')
            return
        if len(detections) in (1, 4, 7, 10, 13) and not self.lock.isChecked():
            self.hero.setText('等待自己出牌')
            self.results.setRowCount(0)
            return
        if manual or self.stable >= 2:
            self.analyze()

    def paint_tiles(self, tiles, confidences=None):
        while self.tile_grid.count():
            widget = self.tile_grid.takeAt(0).widget()
            if widget:
                widget.deleteLater()
        for i, tile in enumerate(tiles):
            button = QPushButton(name(tile))
            button.setToolTip(tile + (f' · 匹配分{confidences[i]:.0%}' if confidences else ' · 手动输入'))
            button.setStyleSheet('padding:7px 2px;color:#f4c56a;' if confidences and confidences[i] < .70 else 'padding:7px 2px;')
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
        return Board(hand=hand, rivers=[parse(e.text()) if e.text().strip() or i in self.river_known or self.confirmed.isChecked() else None
                                      for i, e in enumerate(self.rivers)],
                     indicators=parse(self.dora.text()), melds=melds,
                     riichi=[i for i, c in enumerate(self.reach) if c.isChecked()],
                     scores=[int(x) for x in self.scores.text().split()], dealer=self.dealer.currentIndex(),
                     round_wind=self.wind.currentIndex(), honba=self.honba.value(), sticks=self.sticks.value(),
                     drawn=hand[-1] if hand else None, metadata_confirmed=self.confirmed.isChecked(), hand_number=self.hand_number.value())

    def analyze(self):
        if self.busy & {'analysis', 'load'}:
            return
        try:
            board = self.board()
            signature = repr(board) + str(self.mode.currentIndex())
            if signature == self.last_analyzed:
                return
            self.last_analyzed = signature
            self.paint_tiles(board.hand)
            self.status.setText('正在分析当前手牌…')
            use_model = self.mode.currentIndex() == 1
            self.start('analysis', lambda: self.adviser.recommend(board, use_model))
        except Exception as error:
            self.status.setText(str(error))

    def show_result(self, result):
        top = result['top3'][0]
        self.hero.setText('建议打 ' + top['name'])
        self.engine_label.setText(result['engine'] + ' · ' + datetime.now().strftime('%H:%M:%S'))
        safe = '，对立直家现物' if set(top['safe']) & {i for i, c in enumerate(self.reach) if c.isChecked()} else ''
        self.explanation.setText(f'弃牌后{top["sh"]}向听，{top["ukeire"]}枚可见剩余进张{safe}。')
        self.results.setRowCount(len(result['top3']))
        for row, candidate in enumerate(result['top3']):
            items = [candidate['name'], str(candidate['sh']), str(candidate['ukeire']),
                     f'{candidate["preference"]:.1%}' if 'preference' in candidate else '—']
            for col, value in enumerate(items):
                self.results.setItem(row, col, QTableWidgetItem(value))
        notes = result['notes'] + ['模型偏好分不是胜率。赤五为色彩估计，可点击校正。']
        if result['winning_shape']:
            self.hero.setText('和牌形 · 检查自摸')
            self.explanation.setText('若游戏提供自摸/和牌按钮，请先确认。未满足役等条件时，可参考下方弃牌候选。')
            notes.insert(0, '当前是和牌形：先检查游戏是否提供自摸/和牌按钮。')
        self.notes.setText('\n'.join(notes))
        self.status.setText('建议已更新')

    def load_model(self):
        self.load_button.setEnabled(False)
        self.status.setText('加载Jev中；首次下载底座可能需要几分钟…')
        checkpoint = self.checkpoint.text().strip()
        self.start('load', lambda: self.adviser.load(checkpoint))

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
            'monitor': self.monitor.currentIndex(), 'checkpoint': self.checkpoint.text(),
            'detector': self.detector_path.text()}, ensure_ascii=False, indent=2), encoding='utf-8')

    def load_settings(self):
        try:
            data = json.loads(CONFIG.read_text(encoding='utf-8'))
            self.regions, self.capture_rect = data.get('regions', {}), data.get('capture_rect')
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
    parser.add_argument('--preview', type=Path, help='Render a demo panel to PNG, including headless environments')
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    font_path = Path(os.environ.get('JEV_FONT', ROOT / 'assets/NotoSansSC.ttf'))
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
    app.setStyleSheet(STYLE)
    window = Sidecar(demo=args.demo or bool(args.preview))
    window.show()
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
