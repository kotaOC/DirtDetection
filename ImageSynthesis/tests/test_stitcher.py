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


def test_requires_two_images() -> None:
    with pytest.raises(StitchError, match="2枚以上"):
        ImageStitcher().stitch([make_scene()])


def test_save_and_reload_unicode_path(tmp_path) -> None:
    path = tmp_path / "合成結果.png"
    image = make_scene()
    save_image(path, image)
    loaded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(loaded, image)
