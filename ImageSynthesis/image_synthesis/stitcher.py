"""Feature based registration and feather blending for inspection images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np


class StitchError(RuntimeError):
    """Raised when selected images cannot be registered safely."""


@dataclass(frozen=True)
class StitchProgress:
    current: int
    total: int
    message: str


ProgressCallback = Callable[[StitchProgress], None]


class ImageStitcher:
    """Stitch images in their supplied order using pairwise registration."""

    def __init__(
        self,
        max_features: int = 6000,
        ratio_threshold: float = 0.75,
        min_matches: int = 12,
        ransac_threshold: float = 4.0,
        max_megapixels: float = 120.0,
    ) -> None:
        self.max_features = max_features
        self.ratio_threshold = ratio_threshold
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.max_megapixels = max_megapixels

    def stitch_files(
        self,
        paths: Sequence[str | Path],
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        if len(paths) < 2:
            raise StitchError("合成する画像を2枚以上選択してください。")
        images: list[np.ndarray] = []
        for index, path in enumerate(paths, start=1):
            self._notify(progress, index, len(paths), f"画像を読み込み中: {Path(path).name}")
            # imdecode also supports paths containing Japanese characters on Windows.
            try:
                data = np.fromfile(str(path), dtype=np.uint8)
                image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except OSError as exc:
                raise StitchError(f"画像を読み込めません: {path}") from exc
            if image is None:
                raise StitchError(f"対応していない画像、または破損した画像です: {path}")
            images.append(image)
        return self.stitch(images, progress)

    def stitch(
        self,
        images: Sequence[np.ndarray],
        progress: ProgressCallback | None = None,
    ) -> np.ndarray:
        if len(images) < 2:
            raise StitchError("合成する画像を2枚以上指定してください。")
        normalized = [self._validate_image(image, i) for i, image in enumerate(images)]
        transforms: list[np.ndarray] = [np.eye(3, dtype=np.float64)]

        for index in range(1, len(normalized)):
            self._notify(progress, index, len(normalized) - 1, f"位置合わせ中: {index + 1} / {len(normalized)}")
            current_to_previous = self._estimate_transform(normalized[index - 1], normalized[index])
            transforms.append(transforms[-1] @ current_to_previous)

        self._notify(progress, 0, 1, "合成範囲を計算中")
        translation, size = self._canvas_geometry(normalized, transforms)
        canvas_pixels = size[0] * size[1]
        if canvas_pixels > self.max_megapixels * 1_000_000:
            raise StitchError(
                f"合成結果が大きすぎます ({canvas_pixels / 1_000_000:.1f} MP)。"
                "画像を縮小するか、選択枚数を減らしてください。"
            )
        return self._blend(normalized, [translation @ t for t in transforms], size, progress)

    def _estimate_transform(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, 1600.0 / max(previous.shape[1], current.shape[1]))
        if scale < 1.0:
            previous_work = cv2.resize(previous_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            current_work = cv2.resize(current_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            previous_work, current_work = previous_gray, current_gray

        orb = cv2.ORB_create(nfeatures=self.max_features, fastThreshold=10)
        key_prev, desc_prev = orb.detectAndCompute(previous_work, None)
        key_cur, desc_cur = orb.detectAndCompute(current_work, None)
        if desc_prev is not None and desc_cur is not None:
            matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_cur, desc_prev, k=2)
            good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < self.ratio_threshold * n.distance]
            if len(good) >= self.min_matches:
                src = np.float32([key_cur[m.queryIdx].pt for m in good]) / scale
                dst = np.float32([key_prev[m.trainIdx].pt for m in good]) / scale
                matrix, inliers = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_threshold,
                    maxIters=3000, confidence=0.995,
                )
                inlier_count = int(inliers.sum()) if inliers is not None else 0
                if matrix is not None and inlier_count >= self.min_matches:
                    transform = np.vstack([matrix, [0.0, 0.0, 1.0]])
                    self._check_transform(transform, current.shape)
                    return transform

        return self._phase_correlation_transform(previous_gray, current_gray)

    def _phase_correlation_transform(self, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        height = min(previous.shape[0], current.shape[0])
        width = min(previous.shape[1], current.shape[1])
        prev = cv2.resize(previous, (width, height)).astype(np.float32)
        cur = cv2.resize(current, (width, height)).astype(np.float32)
        window = cv2.createHanningWindow((width, height), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(cur, prev, window)
        if response < 0.08 or abs(dx) > width * 0.95 or abs(dy) > height * 0.5:
            raise StitchError(
                "画像間の重なりを検出できませんでした。撮影順と、十分な重複領域があるか確認してください。"
            )
        scale_x = current.shape[1] / width
        scale_y = current.shape[0] / height
        return np.array([[1.0, 0.0, dx * scale_x], [0.0, 1.0, dy * scale_y], [0.0, 0.0, 1.0]])

    @staticmethod
    def _check_transform(transform: np.ndarray, shape: tuple[int, ...]) -> None:
        linear = transform[:2, :2]
        scale = float(np.sqrt(abs(np.linalg.det(linear))))
        angle = abs(float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0]))))
        dx, dy = transform[0, 2], transform[1, 2]
        if not 0.8 <= scale <= 1.25 or angle > 15 or abs(dy) > shape[0] * 0.6 or abs(dx) > shape[1] * 1.2:
            raise StitchError("不自然な位置合わせ結果を検出しました。画像の順序や重複範囲を確認してください。")

    def _canvas_geometry(
        self, images: Sequence[np.ndarray], transforms: Sequence[np.ndarray]
    ) -> tuple[np.ndarray, tuple[int, int]]:
        all_corners = []
        for image, transform in zip(images, transforms):
            h, w = image.shape[:2]
            corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
            all_corners.append(cv2.perspectiveTransform(corners, transform)[0])
        points = np.concatenate(all_corners)
        minimum = np.floor(points.min(axis=0)).astype(int)
        maximum = np.ceil(points.max(axis=0)).astype(int)
        width, height = (maximum - minimum).tolist()
        translation = np.array([[1.0, 0.0, -minimum[0]], [0.0, 1.0, -minimum[1]], [0.0, 0.0, 1.0]])
        return translation, (int(width), int(height))

    def _blend(
        self,
        images: Sequence[np.ndarray],
        transforms: Sequence[np.ndarray],
        size: tuple[int, int],
        progress: ProgressCallback | None,
    ) -> np.ndarray:
        width, height = size
        accumulator = np.zeros((height, width, 3), dtype=np.float32)
        weight_sum = np.zeros((height, width), dtype=np.float32)
        for index, (image, transform) in enumerate(zip(images, transforms), start=1):
            self._notify(progress, index, len(images), f"画像を合成中: {index} / {len(images)}")
            h, w = image.shape[:2]
            mask = np.full((h, w), 255, dtype=np.uint8)
            # distanceTransform needs an explicit zero-valued exterior. Padding
            # avoids the unbounded values produced by an all-foreground mask.
            padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
            # A small floor keeps the outer edge represented while favoring image centers in overlaps.
            weight = np.maximum(distance, 1.0)
            warped_image = cv2.warpPerspective(image, transform, (width, height), flags=cv2.INTER_LINEAR)
            warped_weight = cv2.warpPerspective(weight, transform, (width, height), flags=cv2.INTER_LINEAR)
            valid = cv2.warpPerspective(mask, transform, (width, height), flags=cv2.INTER_NEAREST) > 0
            warped_weight *= valid
            accumulator += warped_image.astype(np.float32) * warped_weight[..., None]
            weight_sum += warped_weight
        result = accumulator / np.maximum(weight_sum[..., None], 1e-6)
        valid_pixels = weight_sum > 0
        if not valid_pixels.any():
            raise StitchError("合成画像を生成できませんでした。")
        ys, xs = np.where(valid_pixels)
        result = result[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _validate_image(image: np.ndarray, index: int) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise StitchError(f"{index + 1}枚目の画像が無効です。")
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise StitchError(f"{index + 1}枚目の画像形式に対応していません。")
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[2] == 4 else image

    @staticmethod
    def _notify(callback: ProgressCallback | None, current: int, total: int, message: str) -> None:
        if callback:
            callback(StitchProgress(current, total, message))


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an OpenCV image while supporting non-ASCII Windows paths."""
    suffix = Path(path).suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        raise StitchError("保存形式は PNG、JPEG、BMP、TIFF から選択してください。")
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 95] if suffix in {".jpg", ".jpeg"} else []
    success, encoded = cv2.imencode(suffix, image, parameters)
    if not success:
        raise StitchError("画像のエンコードに失敗しました。")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise StitchError(f"画像を保存できません: {path}") from exc
