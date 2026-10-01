import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from image_synthesis.qt_app import MainWindow
from image_synthesis.stitcher import SeamComparison, save_image


def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_initial_action_states() -> None:
    app = qt_app()
    window = MainWindow()
    app.processEvents()
    assert not window.run_button.isEnabled()
    assert not window.save_button.isEnabled()
    window.close()


def test_viewer_keeps_full_resolution_pixmap() -> None:
    app = qt_app()
    window = MainWindow()
    image = np.zeros((180, 2400, 3), dtype=np.uint8)
    window.viewer.set_cv_image(image)
    app.processEvents()
    pixmap = window.viewer._item.pixmap()
    assert (pixmap.width(), pixmap.height()) == (2400, 180)
    window.close()


def test_wide_result_can_be_panned_vertically() -> None:
    app = qt_app()
    window = MainWindow()
    window.resize(1500, 900)
    window.show()
    app.processEvents()
    window.viewer.set_cv_image(np.zeros((180, 2400, 3), dtype=np.uint8))
    app.processEvents()

    vertical_bar = window.viewer.verticalScrollBar()
    assert vertical_bar.maximum() > vertical_bar.minimum()
    window.close()


def test_selects_and_displays_three_views_for_a_seam() -> None:
    app = qt_app()
    window = MainWindow()
    blended = np.zeros((120, 600, 3), dtype=np.uint8)
    comparison = SeamComparison(
        left_source=np.full((120, 160, 3), 40, dtype=np.uint8),
        blended=np.full((120, 160, 3), 110, dtype=np.uint8),
        right_source=np.full((120, 160, 3), 180, dtype=np.uint8),
    )
    outlines = [
        np.float32([[0, 0], [359, 0], [359, 119], [0, 119]]),
        np.float32([[240, 0], [599, 0], [599, 119], [240, 119]]),
    ]
    window._on_completed((blended, [comparison], outlines))

    window._show_seam_comparison()

    left = window.left_source_viewer._item.pixmap().toImage().pixelColor(10, 10)
    center = window.blended_seam_viewer._item.pixmap().toImage().pixelColor(10, 10)
    right = window.right_source_viewer._item.pixmap().toImage().pixelColor(10, 10)
    assert (left.red(), center.red(), right.red()) == (40, 110, 180)
    assert window.preview_stack.currentIndex() == 1
    assert window.seam_selector.count() == 1
    assert not window.seam_button.isEnabled()
    assert window.blended_button.isEnabled()
    window.close()


def test_source_outlines_can_be_toggled() -> None:
    app = qt_app()
    window = MainWindow()
    image = np.zeros((120, 600, 3), dtype=np.uint8)
    outlines = [np.float32([[0, 0], [359, 0], [359, 119], [0, 119]])]
    window.viewer.set_cv_image(image)
    window.viewer.set_source_outlines(outlines)

    window.viewer.set_outlines_visible(False)
    assert all(not item.isVisible() for item in window.viewer._outline_items)
    window.viewer.set_outlines_visible(True)
    assert all(item.isVisible() for item in window.viewer._outline_items)
    window.close()


def test_double_click_opens_source_in_default_photo_app(tmp_path, monkeypatch) -> None:
    app = qt_app()
    window = MainWindow()
    image = np.full((90, 240, 3), 75, dtype=np.uint8)
    path = tmp_path / "source.png"
    save_image(path, image)
    window.paths = [path]
    window._rebuild_list(0)
    opened = []
    monkeypatch.setattr(
        "image_synthesis.qt_app.QDesktopServices.openUrl",
        lambda url: opened.append(url.toLocalFile()) or True,
    )

    window._open_image_external(window.image_list.item(0))
    app.processEvents()

    assert [Path(value) for value in opened] == [path.resolve()]
    window.close()
