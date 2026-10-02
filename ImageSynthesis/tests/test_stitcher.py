import cv2
import numpy as np
import pytest

from image_synthesis.stitcher import ImageStitcher, StitchError, save_image


def make_scene() -> np.ndarray:
    rng = np.random.default_rng(42)
    image = np.full((220, 900, 3), 80, np.uint8)
    image += rng.integers(0, 35, image.shape, dtype=np.uint8)
    for x in range(40, 880, 70):
        cv2.circle(image, (x, 110 + (x % 35)), 18, (200, 170, 120), -1)
        cv2.putText(image, str(x), (x - 20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)
    return image


def test_stitches_overlapping_horizontal_images() -> None:
    scene = make_scene()
    crops = [scene[:, 0:420], scene[:, 250:670], scene[:, 480:900]]
    result = ImageStitcher().stitch(crops)
    assert 860 <= result.shape[1] <= 940
    assert 215 <= result.shape[0] <= 230
    assert np.mean(cv2.absdiff(result[:220, :900], scene)) < 8


def test_produces_three_aligned_views_for_each_seam() -> None:
    scene = make_scene()
    previous = scene[:, :560]
    current = cv2.convertScaleAbs(scene[:, 340:900], alpha=1.0, beta=18)

    blended, comparisons, outlines = ImageStitcher().stitch_with_comparison([previous, current])

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.left_source.shape == comparison.blended.shape
    assert comparison.blended.shape == comparison.right_source.shape
    assert comparison.blended.shape[1] < blended.shape[1]
    assert np.mean(cv2.absdiff(comparison.left_source, comparison.right_source)) > 0.1
    assert 0 <= comparison.center[0] < blended.shape[1]
    assert 0 <= comparison.center[1] < blended.shape[0]
    assert len(outlines) == 2
    assert all(outline.shape == (4, 2) for outline in outlines)


def test_uses_previous_right_and_current_left_for_registration() -> None:
    scene = make_scene()
    previous = scene[:, :640].copy()
    current = scene[:, 460:900].copy()
    current = cv2.copyMakeBorder(current, 0, 0, 0, 200, cv2.BORDER_CONSTANT, value=(35, 35, 35))

    # A strong camera-fixed overlay creates many zero-motion features.  These
    # must not override the actual overlap at the facing halves of the images.
    for image in (previous, current):
        cv2.rectangle(image, (210, 20), (430, 65), (0, 0, 0), -1)
        cv2.circle(image, (320, 145), 38, (0, 0, 0), -1)

    transform = ImageStitcher()._estimate_transform(previous, current)

    assert transform[0, 2] == pytest.approx(460, abs=4)
    assert transform[1, 2] == pytest.approx(0, abs=4)


def test_requires_two_images() -> None:
    with pytest.raises(StitchError, match="2枚以上"):
        ImageStitcher().stitch([make_scene()])


def test_registration_error_identifies_image_pair(monkeypatch) -> None:
    stitcher = ImageStitcher()
    calls = 0

    def estimate(_previous, _current):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise StitchError("十分な重なりを検出できませんでした。")
        return np.eye(3, dtype=np.float64)

    monkeypatch.setattr(stitcher, "_estimate_transform", estimate)
    with pytest.raises(StitchError, match=r"継ぎ目 2（画像 2 と画像 3 の間）") as error:
        stitcher.stitch([make_scene(), make_scene(), make_scene()])
    partial_result, comparisons, outlines = error.value.partial_output
    assert partial_result.shape == make_scene().shape
    assert len(comparisons) == 1
    assert len(outlines) == 2


def test_save_and_reload_unicode_path(tmp_path) -> None:
    path = tmp_path / "合成結果.png"
    image = make_scene()
    save_image(path, image)
    loaded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(loaded, image)
