"""High-DPI Qt front end for the image synthesis engine."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPoint, QPointF, QSize, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImage, QImageReader, QPainter, QPen, QPixmap, QPolygonF, QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSizePolicy, QSpacerItem, QSplitter,
    QStackedWidget, QStatusBar, QVBoxLayout, QWidget,
)

from .stitcher import ImageStitcher, SeamComparison, StitchError, StitchProgress, save_image


def enable_windows_high_dpi() -> None:
    """Declare Per-Monitor V2 awareness before QApplication is constructed."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except (AttributeError, OSError):
                pass


COLORS = {
    "bg": "#07111F", "panel": "#0C192A", "card": "#112239", "card2": "#0A1727",
    "line": "#243C57", "line2": "#345A7D", "text": "#F2F7FC", "muted": "#8EA5BC",
    "blue": "#2189D8", "blue2": "#36A2ED", "green": "#25A96B", "green2": "#31C47C",
    "red": "#B94B5E", "red2": "#CE5B6D", "amber": "#E1A93B",
}


STYLESHEET = f"""
* {{ font-family: 'Segoe UI', 'Yu Gothic UI'; font-size: 10pt; color: {COLORS['text']}; }}
QMainWindow, QWidget#root {{ background: {COLORS['bg']}; }}
QFrame#panel {{ background: {COLORS['panel']}; border: 1px solid {COLORS['line']}; border-radius: 10px; }}
QFrame#footer {{ background: {COLORS['panel']}; border-top: 1px solid {COLORS['line']}; }}
QLabel#eyebrow {{ color: #56B9F3; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }}
QLabel#title {{ font-size: 23pt; font-weight: 650; }}
QLabel#subtitle, QLabel#muted {{ color: {COLORS['muted']}; }}
QLabel#section {{ font-size: 12pt; font-weight: 650; }}
QLabel#count {{ color: #60C5FF; font: 700 20pt 'Segoe UI'; }}
QLabel#mono {{ color: {COLORS['muted']}; font: 9pt 'Consolas'; }}
QPushButton {{ min-height: 40px; padding: 0 16px; border: 1px solid {COLORS['line']}; border-radius: 7px;
               background: {COLORS['card']}; font-weight: 600; }}
QPushButton:hover {{ background: #18304A; border-color: {COLORS['line2']}; }}
QPushButton:pressed {{ background: #0D1D30; }}
QPushButton:disabled {{ color: #607890; background: #132438; border-color: #20354B; }}
QPushButton#primary {{ background: {COLORS['blue']}; border-color: {COLORS['blue']}; font-size: 11pt; min-height: 48px; }}
QPushButton#primary:hover {{ background: {COLORS['blue2']}; }}
QPushButton#success {{ background: {COLORS['green']}; border-color: {COLORS['green']}; min-height: 48px; }}
QPushButton#success:hover {{ background: {COLORS['green2']}; }}
QPushButton#danger {{ color: #FFDDE2; background: #512636; border-color: #713346; }}
QPushButton#danger:hover {{ background: {COLORS['red']}; }}
QPushButton#tool {{ min-width: 42px; min-height: 34px; padding: 0 10px; }}
QPushButton#tool:checked {{ background: {COLORS['blue']}; border-color: {COLORS['blue2']}; }}
QComboBox {{ min-height: 34px; padding: 0 10px; border: 1px solid {COLORS['line']}; border-radius: 7px;
             background: {COLORS['card']}; selection-background-color: {COLORS['blue']}; }}
QComboBox::drop-down {{ border: 0; width: 24px; }}
QComboBox QAbstractItemView {{ background: {COLORS['card']}; border: 1px solid {COLORS['line2']}; }}
QListWidget {{ background: {COLORS['card2']}; border: 1px solid {COLORS['line']}; border-radius: 7px;
               outline: none; padding: 7px; }}
QListWidget::item {{ border: 1px solid {COLORS['line']}; border-radius: 7px; margin: 4px; background: {COLORS['card']}; }}
QListWidget::item:hover {{ background: #152B44; border-color: {COLORS['line2']}; }}
QListWidget::item:selected {{ background: #153451; border: 2px solid {COLORS['blue']}; }}
QProgressBar {{ min-height: 8px; max-height: 8px; border: 0; border-radius: 4px; background: #071321; }}
QProgressBar::chunk {{ border-radius: 4px; background: {COLORS['blue']}; }}
QScrollBar:vertical {{ background: #081523; width: 12px; margin: 0; }}
QScrollBar:horizontal {{ background: #081523; height: 12px; margin: 0; }}
QScrollBar::handle {{ background: #2A4967; border-radius: 5px; min-width: 28px; min-height: 28px; }}
QScrollBar::handle:hover {{ background: #39668E; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QSplitter::handle {{ background: {COLORS['bg']}; width: 14px; }}
"""


class StitchWorker(QObject):
    progress = Signal(float, str)
    completed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, paths: list[Path]) -> None:
        super().__init__()
        self.paths = paths

    @Slot()
    def run(self) -> None:
        try:
            result = ImageStitcher().stitch_files_with_comparison(self.paths, self._progress)
            self.completed.emit(result)
        except Exception as exc:
            prefix = "" if isinstance(exc, StitchError) else "予期しないエラー: "
            self.failed.emit(prefix + str(exc))
        finally:
            self.finished.emit()

    def _progress(self, update: StitchProgress) -> None:
        stage = update.message
        fraction = update.current / max(update.total, 1)
        if stage.startswith("画像を読み込み中"):
            value = 20 * fraction
        elif stage.startswith("位置合わせ中"):
            value = 20 + 45 * fraction
        elif stage.startswith("合成範囲"):
            value = 68
        elif stage.startswith("画像を合成中"):
            value = 70 + 30 * fraction
        else:
            value = 0
        self.progress.emit(value, stage)


class ImageViewer(QGraphicsView):
    """Full-resolution pixmap viewer with fit, 100%, zoom and hand panning."""

    zoomChanged = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setScene(self._scene)
        self.setBackgroundBrush(QColor("#020811"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform | QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._has_image = False
        self._fit = True
        self._zoom = 1.0
        self._outline_items: list[object] = []

    def set_cv_image(self, image: np.ndarray, preserve_view: bool = False) -> None:
        old_center = self.mapToScene(self.viewport().rect().center())
        old_transform = self.transform()
        had_image = self._has_image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        qimage = QImage(rgb.data, w, h, channels * w, QImage.Format.Format_RGB888).copy()
        self._item.setPixmap(QPixmap.fromImage(qimage))
        self._has_image = True
        if preserve_view and had_image:
            self.setTransform(old_transform)
            self._zoom = self.transform().m11()
            self._update_pan_area()
            self.centerOn(old_center)
            self.zoomChanged.emit(round(self._zoom * 100))
        else:
            self.fit_to_window()

    def clear_image(self) -> None:
        self.set_source_outlines([])
        self._item.setPixmap(QPixmap())
        self._has_image = False
        self.resetTransform()
        self.viewport().update()

    def set_source_outlines(self, outlines: Sequence[np.ndarray]) -> None:
        for item in self._outline_items:
            self._scene.removeItem(item)  # type: ignore[arg-type]
        self._outline_items.clear()
        colors = ["#FF4D6D", "#35C2FF", "#FFD166", "#45D483", "#C77DFF", "#FF8C42"]
        for index, points in enumerate(outlines):
            color = QColor(colors[index % len(colors)])
            polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in points])
            pen = QPen(color, 3.0)
            pen.setCosmetic(True)
            outline_item = self._scene.addPolygon(polygon, pen)
            outline_item.setZValue(10)
            label_item = self._scene.addSimpleText(str(index + 1), QFont("Segoe UI", 14, QFont.Weight.Bold))
            label_item.setBrush(color)
            label_item.setPos(polygon[0] + QPointF(7, 5))
            label_item.setZValue(11)
            self._outline_items.extend((outline_item, label_item))

    def set_outlines_visible(self, visible: bool) -> None:
        for item in self._outline_items:
            item.setVisible(visible)  # type: ignore[attr-defined]

    def fit_to_window(self) -> None:
        if not self._has_image:
            return
        self._fit = True
        self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)
        self._update_zoom_from_transform()
        self._update_pan_area()

    def actual_size(self) -> None:
        if not self._has_image:
            return
        self._fit = False
        self.resetTransform()
        self._zoom = 1.0
        self._update_pan_area()
        self.zoomChanged.emit(100)

    def zoom_by(self, factor: float) -> None:
        if not self._has_image:
            return
        target = self._zoom * factor
        if not 0.05 <= target <= 8.0:
            return
        self._fit = False
        self.scale(factor, factor)
        self._zoom = target
        self._update_pan_area()
        self.zoomChanged.emit(round(self._zoom * 100))

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._has_image:
            self.zoom_by(1.18 if event.angleDelta().y() > 0 else 1 / 1.18)
            event.accept()
        else:
            super().wheelEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit and self._has_image:
            self.fit_to_window()
        elif self._has_image:
            self._update_pan_area()

    def drawForeground(self, painter: QPainter, rect) -> None:
        super().drawForeground(painter, rect)
        if self._has_image:
            return
        painter.save()
        painter.resetTransform()
        viewport = self.viewport().rect()
        painter.setPen(QColor(COLORS["line2"]))
        painter.setFont(QFont("Yu Gothic UI", 12, QFont.Weight.DemiBold))
        painter.drawText(viewport, Qt.AlignmentFlag.AlignCenter, "画像を2枚以上追加して\n合成を開始してください")
        painter.restore()

    def _update_zoom_from_transform(self) -> None:
        self._zoom = self.transform().m11()
        self.zoomChanged.emit(round(self._zoom * 100))

    def _update_pan_area(self) -> None:
        """Add viewport-sized space around the image for unrestricted panning."""
        if not self._has_image:
            return
        scale = max(self.transform().m11(), 1e-6)
        horizontal_margin = self.viewport().width() / (2.0 * scale)
        vertical_margin = self.viewport().height() / (2.0 * scale)
        bounds = self._item.boundingRect()
        self._scene.setSceneRect(
            bounds.adjusted(
                -horizontal_margin,
                -vertical_margin,
                horizontal_margin,
                vertical_margin,
            )
        )


class InputImageWidget(QWidget):
    def __init__(self, path: Path, number: int, thumbnail: QPixmap) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(7, 4, 9, 4)
        layout.setSpacing(8)
        order = QLabel(f"{number:02d}")
        order.setStyleSheet("color:#61C7FF;font:700 14pt 'Consolas';min-width:30px;")
        layout.addWidget(order)
        preview = QLabel()
        preview.setFixedSize(86, 52)
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setStyleSheet("background:#030A12;border-radius:4px;")
        preview.setPixmap(thumbnail)
        layout.addWidget(preview)
        details = QVBoxLayout()
        details.setSpacing(4)
        filename = QLabel(path.name)
        filename.setToolTip(str(path))
        filename.setStyleSheet("font-weight:600;")
        filename.setWordWrap(True)
        details.addWidget(filename)
        dimensions = QLabel(self._dimensions(path))
        dimensions.setObjectName("muted")
        details.addWidget(dimensions)
        details.addStretch()
        layout.addLayout(details, 1)

    @staticmethod
    def _dimensions(path: Path) -> str:
        reader = QImageReader(str(path))
        size = reader.size()
        return f"{size.width():,} × {size.height():,} px" if size.isValid() else "サイズ不明"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[Path] = []
        self.last_removed: tuple[Path, int] | None = None
        self.result: np.ndarray | None = None
        self.seam_comparisons: list[SeamComparison] = []
        self.source_outlines: list[np.ndarray] = []
        self.last_error: str | None = None
        self.thread: QThread | None = None
        self.worker: StitchWorker | None = None
        self.setWindowTitle("Image Synthesis | 金属部品 画像合成")
        self.resize(1500, 900)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(STYLESHEET)
        self._build_ui()
        self._update_actions()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 14, 20, 12)
        outer.setSpacing(10)
        outer.addWidget(self._header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._input_panel())

        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right_layout.addWidget(self._preview_panel(), 1)
        right_layout.addWidget(self._footer())
        splitter.addWidget(right_column)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 1040])
        outer.addWidget(splitter, 1)

    def _header(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(2, 0, 2, 0)
        titles = QVBoxLayout()
        eyebrow = QLabel("IMAGE PROCESSING  /  STITCHING")
        eyebrow.setObjectName("eyebrow")
        titles.addWidget(eyebrow)
        title = QLabel("Image Synthesis")
        title.setObjectName("title")
        titles.addWidget(title)
        subtitle = QLabel("金属部品の連続撮影画像を、高精度に1枚へ合成")
        subtitle.setObjectName("subtitle")
        titles.addWidget(subtitle)
        layout.addLayout(titles)
        layout.addStretch()
        self.ready_dot = QLabel("●")
        self.ready_dot.setStyleSheet(f"color:{COLORS['muted']};font-size:12pt;")
        layout.addWidget(self.ready_dot)
        self.ready_label = QLabel("Ready")
        self.ready_label.setStyleSheet("font-weight:650;")
        layout.addWidget(self.ready_label)
        return widget

    def _panel_header(self, number: str, title: str, subtitle: str) -> tuple[QWidget, QHBoxLayout]:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        badge = QLabel(number)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(34, 28)
        badge.setStyleSheet(f"background:{COLORS['blue']};border-radius:5px;font:700 9pt 'Consolas';")
        layout.addWidget(badge)
        text = QVBoxLayout()
        text.setSpacing(1)
        heading = QLabel(title)
        heading.setObjectName("section")
        text.addWidget(heading)
        hint = QLabel(subtitle)
        hint.setObjectName("muted")
        text.addWidget(hint)
        layout.addLayout(text)
        layout.addStretch()
        return widget, layout

    def _input_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        panel.setMinimumWidth(370)
        panel.setMaximumWidth(470)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 13, 14, 13)
        layout.setSpacing(6)
        header, header_layout = self._panel_header("01", "INPUT IMAGES", "入力画像・撮影順序")
        self.count_label = QLabel("0")
        self.count_label.setObjectName("count")
        header_layout.addWidget(self.count_label)
        unit = QLabel(" IMAGES")
        unit.setObjectName("muted")
        header_layout.addWidget(unit)
        layout.addWidget(header)

        self.empty_label = QLabel("画像が追加されていません  —  PNG / JPEG / BMP / TIFF")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setObjectName("muted")
        self.empty_label.setFixedHeight(24)
        layout.addWidget(self.empty_label)

        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        # Keep enough viewport space for at least four compact image rows.
        self.image_list.setMinimumHeight(280)
        self.image_list.currentRowChanged.connect(self._update_actions)
        self.image_list.itemDoubleClicked.connect(self._open_image_external)
        layout.addWidget(self.image_list, 1)

        self.add_button = QPushButton("＋  画像を追加")
        self.add_button.setObjectName("primary")
        self.add_button.setFixedHeight(38)
        self.add_button.clicked.connect(self._add_images)
        layout.addWidget(self.add_button)
        row = QHBoxLayout()
        self.up_button = QPushButton("↑  上へ")
        self.down_button = QPushButton("↓  下へ")
        self.up_button.setFixedHeight(32)
        self.down_button.setFixedHeight(32)
        self.up_button.clicked.connect(lambda: self._move(-1))
        self.down_button.clicked.connect(lambda: self._move(1))
        row.addWidget(self.up_button)
        row.addWidget(self.down_button)
        layout.addLayout(row)
        danger = QHBoxLayout()
        self.remove_button = QPushButton("選択画像を削除")
        self.remove_button.setObjectName("danger")
        self.remove_button.setFixedHeight(32)
        self.remove_button.clicked.connect(self._remove_selected)
        self.undo_remove_button = QPushButton("↶  元に戻す")
        self.undo_remove_button.setFixedHeight(32)
        self.undo_remove_button.setToolTip("直前に削除した画像を元の位置へ戻す")
        self.undo_remove_button.clicked.connect(self._undo_remove)
        self.clear_button = QPushButton("すべてクリア")
        self.clear_button.setFixedHeight(32)
        self.clear_button.clicked.connect(self._clear_all)
        danger.addWidget(self.remove_button)
        danger.addWidget(self.undo_remove_button)
        danger.addWidget(self.clear_button)
        layout.addLayout(danger)
        return panel

    def _preview_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 17, 18, 17)
        layout.setSpacing(11)
        header, header_layout = self._panel_header("02", "STITCHED PREVIEW", "合成結果プレビュー")
        self.result_size = QLabel("NO RESULT")
        self.result_size.setObjectName("mono")
        header_layout.addWidget(self.result_size)
        layout.addWidget(header)
        tool_rows = QVBoxLayout()
        tool_rows.setSpacing(7)
        view_tools = QHBoxLayout()
        compare_tools = QHBoxLayout()
        fit = QPushButton("画面に合わせる")
        fit.setObjectName("tool")
        fit.setMinimumWidth(118)
        fit.clicked.connect(self._fit)
        actual = QPushButton("100%")
        actual.setObjectName("tool")
        actual.clicked.connect(self._actual)
        self.blended_button = QPushButton("全体表示")
        self.blended_button.setObjectName("tool")
        self.blended_button.setMinimumWidth(90)
        self.blended_button.setToolTip("合成後の画像全体を表示")
        self.blended_button.clicked.connect(self._show_blended)
        self.seam_button = QPushButton("3画面比較")
        self.seam_button.setObjectName("tool")
        self.seam_button.setMinimumWidth(108)
        self.seam_button.setToolTip("左右の元画像と合成後の継ぎ目を並べて表示")
        self.seam_button.clicked.connect(self._show_seam_comparison)
        self.seam_selector = QComboBox()
        self.seam_selector.setMinimumWidth(230)
        self.seam_selector.setToolTip("比較する合成部分を選択")
        self.seam_selector.currentIndexChanged.connect(self._update_seam_comparison)
        self.outline_button = QPushButton("画像範囲 OFF")
        self.outline_button.setObjectName("tool")
        self.outline_button.setCheckable(True)
        self.outline_button.setMinimumWidth(112)
        self.outline_button.setToolTip("各元画像が合成結果のどこに配置されたかを表示")
        self.outline_button.toggled.connect(self._toggle_outlines)
        zoom_out = QPushButton("−")
        zoom_out.setObjectName("tool")
        zoom_out.clicked.connect(lambda: self._zoom_by(1 / 1.25))
        zoom_in = QPushButton("＋")
        zoom_in.setObjectName("tool")
        zoom_in.clicked.connect(lambda: self._zoom_by(1.25))
        self.zoom_label = QLabel("FIT")
        self.zoom_label.setStyleSheet("color:#61C7FF;font:10pt 'Consolas';min-width:54px;")
        for control in (fit, actual, zoom_out, zoom_in, self.zoom_label):
            view_tools.addWidget(control)
        view_tools.addStretch()
        hint = QLabel("ホイール: ズーム  /  ドラッグ: パン")
        hint.setObjectName("muted")
        view_tools.addWidget(hint)

        compare_label = QLabel("表示モード / 比較箇所:")
        compare_label.setObjectName("muted")
        compare_tools.addWidget(compare_label)
        compare_tools.addWidget(self.blended_button)
        compare_tools.addWidget(self.seam_button)
        compare_tools.addWidget(self.outline_button)
        compare_tools.addWidget(self.seam_selector)
        compare_tools.addStretch()
        tool_rows.addLayout(view_tools)
        tool_rows.addLayout(compare_tools)
        layout.addLayout(tool_rows)
        self.viewer = ImageViewer()
        self.viewer.zoomChanged.connect(lambda value: self.zoom_label.setText(f"{value}%"))
        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.viewer)

        comparison = QWidget()
        comparison_layout = QHBoxLayout(comparison)
        comparison_layout.setContentsMargins(0, 0, 0, 0)
        comparison_layout.setSpacing(8)
        self.left_source_viewer = ImageViewer()
        self.blended_seam_viewer = ImageViewer()
        self.right_source_viewer = ImageViewer()
        comparison_layout.addWidget(self._comparison_column("左側の元画像（右端）", self.left_source_viewer))
        comparison_layout.addWidget(self._comparison_column("合成後", self.blended_seam_viewer))
        comparison_layout.addWidget(self._comparison_column("右側の元画像（左端）", self.right_source_viewer))
        self.preview_stack.addWidget(comparison)
        layout.addWidget(self.preview_stack, 1)
        return panel

    @staticmethod
    def _comparison_column(title: str, viewer: ImageViewer) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-weight:650;color:#B9CCE0;")
        layout.addWidget(label)
        layout.addWidget(viewer, 1)
        return column

    def _footer(self) -> QWidget:
        footer = QFrame()
        footer.setObjectName("panel")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(10)
        state = QVBoxLayout()
        state.setSpacing(6)
        status_row = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color:{COLORS['muted']};")
        self.status_text = QLabel("Ready — 画像を2枚以上追加してください")
        self.progress_value = QLabel("0%")
        self.progress_value.setObjectName("mono")
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_text)
        status_row.addStretch()
        status_row.addWidget(self.progress_value)
        state.addLayout(status_row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        state.addWidget(self.progress)
        layout.addLayout(state, 1)
        self.save_button = QPushButton("結果を保存")
        self.save_button.setObjectName("success")
        self.save_button.setFixedHeight(44)
        self.save_button.clicked.connect(self._save)
        layout.addWidget(self.save_button)
        self.run_button = QPushButton("合成を開始  ▶")
        self.run_button.setObjectName("primary")
        self.run_button.setFixedHeight(44)
        self.run_button.setMinimumWidth(190)
        self.run_button.clicked.connect(self._start)
        layout.addWidget(self.run_button)
        return footer

    def _add_images(self) -> None:
        values, _ = QFileDialog.getOpenFileNames(self, "合成する画像を選択", "", "画像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;すべて (*.*)")
        known = {str(path.resolve()) for path in self.paths}
        for value in values:
            path = Path(value)
            if str(path.resolve()) not in known:
                self.paths.append(path)
                known.add(str(path.resolve()))
        self._rebuild_list(len(self.paths) - 1 if values else self.image_list.currentRow())

    def _thumbnail(self, path: Path) -> QPixmap:
        source = QPixmap(str(path))
        if source.isNull():
            source = QPixmap(112, 72)
            source.fill(QColor("#07111F"))
            return source
        ratio = self.devicePixelRatioF()
        scaled = source.scaled(round(86 * ratio), round(52 * ratio), Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        scaled.setDevicePixelRatio(ratio)
        return scaled

    def _rebuild_list(self, selected: int = -1) -> None:
        self.image_list.clear()
        for number, path in enumerate(self.paths, 1):
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setSizeHint(QSize(330, 66))
            self.image_list.addItem(item)
            self.image_list.setItemWidget(item, InputImageWidget(path, number, self._thumbnail(path)))
        self.count_label.setText(str(len(self.paths)))
        self.empty_label.setVisible(not self.paths)
        if self.paths and selected >= 0:
            self.image_list.setCurrentRow(min(selected, len(self.paths) - 1))
        self._update_actions()

    @Slot(QListWidgetItem)
    def _open_image_external(self, item: QListWidgetItem) -> None:
        path_text = item.data(Qt.ItemDataRole.UserRole)
        if not path_text:
            return
        path = Path(path_text)
        if not path.is_file() or not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve()))):
            QMessageBox.warning(self, "画像表示エラー", f"既定のフォトアプリで画像を開けません: {path}")

    def _move(self, offset: int) -> None:
        current = self.image_list.currentRow()
        target = current + offset
        if current < 0 or not 0 <= target < len(self.paths):
            return
        self.paths[current], self.paths[target] = self.paths[target], self.paths[current]
        self._rebuild_list(target)

    def _remove_selected(self) -> None:
        current = self.image_list.currentRow()
        if current < 0:
            return
        removed = self.paths.pop(current)
        self.last_removed = (removed, current)
        self._rebuild_list(min(current, len(self.paths) - 1))

    def _undo_remove(self) -> None:
        if self.last_removed is None:
            return
        path, original_index = self.last_removed
        self.last_removed = None
        if path in self.paths:
            self._rebuild_list(self.paths.index(path))
            return
        restored_index = min(original_index, len(self.paths))
        self.paths.insert(restored_index, path)
        self._rebuild_list(restored_index)

    def _clear_all(self) -> None:
        if not self.paths or not self._confirm_clear():
            return
        self.paths.clear()
        self.last_removed = None
        self.result = None
        self.seam_comparisons = []
        self.source_outlines = []
        self.last_error = None
        self.seam_selector.clear()
        self.viewer.clear_image()
        self.left_source_viewer.clear_image()
        self.blended_seam_viewer.clear_image()
        self.right_source_viewer.clear_image()
        self.result_size.setText("NO RESULT")
        self.progress.setValue(0)
        self.progress_value.setText("0%")
        self._rebuild_list()

    def _confirm_clear(self) -> bool:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("入力画像をクリア")
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setText("追加した画像をすべて取り除きますか？")
        dialog.setInformativeText("合成結果と比較表示もクリアされます。")
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.No)
        yes_button = dialog.button(QMessageBox.StandardButton.Yes)
        no_button = dialog.button(QMessageBox.StandardButton.No)
        if yes_button is not None:
            yes_button.setText("クリア")
            yes_button.setObjectName("clearConfirm")
        if no_button is not None:
            no_button.setText("キャンセル")
            no_button.setObjectName("cancelConfirm")
        dialog.setStyleSheet(self._message_dialog_stylesheet())
        return dialog.exec() == QMessageBox.StandardButton.Yes

    def _update_actions(self, *_args) -> None:
        busy = self.thread is not None and self.thread.isRunning()
        row = self.image_list.currentRow() if hasattr(self, "image_list") else -1
        self.run_button.setEnabled(len(self.paths) >= 2 and not busy)
        self.save_button.setEnabled(self.result is not None and not busy)
        self.add_button.setEnabled(not busy)
        self.remove_button.setEnabled(row >= 0 and not busy)
        self.undo_remove_button.setEnabled(self.last_removed is not None and not busy)
        self.clear_button.setEnabled(bool(self.paths) and not busy)
        self.up_button.setEnabled(row > 0 and not busy)
        self.down_button.setEnabled(0 <= row < len(self.paths) - 1 and not busy)
        comparison_available = self.result is not None and bool(self.seam_comparisons)
        if not comparison_available:
            self.blended_button.setEnabled(False)
            self.seam_button.setEnabled(False)
            self.seam_selector.setEnabled(False)
        self.outline_button.setEnabled(self.result is not None)
        if not busy:
            if self.last_error:
                self._set_status("Error — 合成に失敗しました。詳細はエラー画面を確認してください。", "error")
                self.status_text.setToolTip(self.last_error)
            else:
                self._set_status("Ready" if len(self.paths) >= 2 else "Ready — 画像を2枚以上追加してください", "ready")
                self.status_text.setToolTip("")

    def _start(self) -> None:
        if len(self.paths) < 2:
            return
        self.result = None
        self.seam_comparisons = []
        self.source_outlines = []
        self.last_error = None
        self.seam_selector.clear()
        self.progress.setValue(0)
        self.save_button.setEnabled(False)
        self._set_status("画像を読み込み中...", "working")
        self.status_text.setToolTip("")
        self.thread = QThread(self)
        self.worker = StitchWorker(list(self.paths))
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()
        self._update_actions()

    @Slot(float, str)
    def _on_progress(self, value: float, message: str) -> None:
        value_int = max(0, min(100, round(value)))
        self.progress.setValue(value_int)
        self.progress_value.setText(f"{value_int}%")
        self._set_status(message, "working")

    @Slot(object)
    def _on_completed(self, result: object) -> None:
        self.last_error = None
        self.result, self.seam_comparisons, self.source_outlines = result  # type: ignore[misc]
        height, width = self.result.shape[:2]
        self.viewer.set_cv_image(self.result)
        self.viewer.set_source_outlines(self.source_outlines)
        self.viewer.set_outlines_visible(self.outline_button.isChecked())
        self.preview_stack.setCurrentIndex(0)
        self.seam_selector.blockSignals(True)
        self.seam_selector.clear()
        for index in range(len(self.seam_comparisons)):
            self.seam_selector.addItem(
                f"継ぎ目 {index + 1}: 画像 {index + 1} ↔ {index + 2}"
            )
        self.seam_selector.blockSignals(False)
        if self.seam_comparisons:
            self.seam_selector.setCurrentIndex(0)
            self._update_seam_comparison(0)
        self._set_comparison_buttons("blended")
        self.result_size.setText(f"{width:,} × {height:,} PX")
        self.progress.setValue(100)
        self.progress_value.setText("100%")
        self._set_status("Completed — 合成が完了しました", "success")
        self.status_text.setToolTip("")

    @Slot(str)
    def _on_failed(self, text: str) -> None:
        self.last_error = text
        self.progress_value.setText("ERROR")
        self._set_status("Error — 合成に失敗しました。詳細はエラー画面を確認してください。", "error")
        self.status_text.setToolTip(text)
        self._show_error_dialog(text)

    def _show_error_dialog(self, text: str) -> None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("合成エラー")
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setText("画像を合成できませんでした。")
        dialog.setInformativeText(text)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.setStyleSheet(self._message_dialog_stylesheet())
        dialog.exec()

    @staticmethod
    def _message_dialog_stylesheet() -> str:
        return f"""
            QMessageBox {{ background: {COLORS['panel']}; }}
            QMessageBox QLabel {{ color: {COLORS['text']}; font-size: 11pt; }}
            QMessageBox QLabel#qt_msgbox_label,
            QMessageBox QLabel#qt_msgbox_informativelabel {{ min-width: 600px; }}
            QMessageBox QPushButton {{ min-width: 90px; min-height: 38px; background: {COLORS['card']};
                                      color: {COLORS['text']}; border: 1px solid {COLORS['line2']}; }}
            QMessageBox QPushButton:hover {{ background: #18304A; }}
            QMessageBox QPushButton#clearConfirm {{ background: {COLORS['red']}; border-color: {COLORS['red2']}; }}
            QMessageBox QPushButton#clearConfirm:hover {{ background: {COLORS['red2']}; }}
            QMessageBox QPushButton#cancelConfirm {{ background: {COLORS['blue']}; border-color: {COLORS['blue2']}; }}
            QMessageBox QPushButton#cancelConfirm:hover {{ background: {COLORS['blue2']}; }}
            """

    @Slot()
    def _thread_finished(self) -> None:
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = None
        self.worker = None
        self._update_actions()
        if self.result is not None:
            self._set_status("Completed — 合成が完了しました", "success")

    def _set_status(self, text: str, state: str) -> None:
        color = {"ready": COLORS["muted"], "working": COLORS["amber"], "success": COLORS["green2"], "error": "#F06B78"}[state]
        self.status_text.setText(text)
        self.status_dot.setStyleSheet(f"color:{color};")
        self.ready_dot.setStyleSheet(f"color:{color};font-size:12pt;")
        self.ready_label.setText({"ready": "Ready", "working": "Processing", "success": "Completed", "error": "Error"}[state])

    def _fit(self) -> None:
        for viewer in self._active_viewers():
            viewer.fit_to_window()
        self.zoom_label.setText("FIT")

    def _actual(self) -> None:
        for viewer in self._active_viewers():
            viewer.actual_size()

    def _zoom_by(self, factor: float) -> None:
        for viewer in self._active_viewers():
            viewer.zoom_by(factor)

    def _active_viewers(self) -> tuple[ImageViewer, ...]:
        if self.preview_stack.currentIndex() == 0:
            return (self.viewer,)
        return (self.left_source_viewer, self.blended_seam_viewer, self.right_source_viewer)

    def _show_blended(self) -> None:
        if self.result is None:
            return
        self.preview_stack.setCurrentIndex(0)
        self._set_comparison_buttons("blended")

    def _show_seam_comparison(self) -> None:
        if not self.seam_comparisons:
            return
        self._update_seam_comparison(self.seam_selector.currentIndex())
        self.preview_stack.setCurrentIndex(1)
        self._set_comparison_buttons("seam")

    @Slot(int)
    def _update_seam_comparison(self, index: int) -> None:
        if not 0 <= index < len(self.seam_comparisons):
            return
        comparison = self.seam_comparisons[index]
        self.left_source_viewer.set_cv_image(comparison.left_source)
        self.blended_seam_viewer.set_cv_image(comparison.blended)
        self.right_source_viewer.set_cv_image(comparison.right_source)

    def _set_comparison_buttons(self, active: str) -> None:
        available = self.result is not None and bool(self.seam_comparisons)
        self.blended_button.setEnabled(available and active != "blended")
        self.seam_button.setEnabled(available and active != "seam")
        self.seam_selector.setEnabled(available)

    @Slot(bool)
    def _toggle_outlines(self, visible: bool) -> None:
        self.viewer.set_outlines_visible(visible)
        self.outline_button.setText("画像範囲 ON" if visible else "画像範囲 OFF")
        if visible and self.result is not None:
            self._show_blended()

    def _save(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "合成画像を保存", "synthesized.png", "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff);;BMP (*.bmp)")
        if not path:
            return
        try:
            save_image(path, self.result)
            self._set_status(f"Completed — 保存しました: {Path(path).name}", "success")
        except StitchError as exc:
            QMessageBox.critical(self, "保存エラー", str(exc))


def main() -> None:
    enable_windows_high_dpi()
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setApplicationName("Image Synthesis")
    app.setStyle("Fusion")
    window = MainWindow()
    # Start in the same maximized state as pressing the title bar's □ button.
    window.showMaximized()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
