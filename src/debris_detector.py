from enum import Enum

import cv2
import numpy as np

from src.config import MAX_CONTOUR_AREA, MIN_CONTOUR_AREA, MORPH_KERNEL_SIZE
from src.debris_segmenter import DebrisSegmenter


class ThresholdMode(Enum):
    MEDIAPIPE = "mediapipe"
    AUTO = "auto"
    LIGHT_BG = "claro"
    DARK_BG = "escuro"
    ADAPTIVE = "adaptativo"


class DebrisDetector:
    def __init__(self):
        self.mode = ThresholdMode.MEDIAPIPE
        self._segmenter: DebrisSegmenter | None = None

    def close(self):
        if self._segmenter is not None:
            self._segmenter.close()
            self._segmenter = None

    def cycle_mode(self) -> ThresholdMode:
        modes = list(ThresholdMode)
        idx = (modes.index(self.mode) + 1) % len(modes)
        self.mode = modes[idx]
        return self.mode

    def _get_segmenter(self) -> DebrisSegmenter:
        if self._segmenter is None:
            self._segmenter = DebrisSegmenter()
        return self._segmenter

    def _min_area(self, frame: np.ndarray) -> int:
        frame_area = frame.shape[0] * frame.shape[1]
        ref_area = 640 * 480
        return max(200, int(MIN_CONTOUR_AREA * frame_area / ref_area))

    def _max_area(self, frame: np.ndarray) -> int:
        frame_area = frame.shape[0] * frame.shape[1]
        ref_area = 640 * 480
        return int(MAX_CONTOUR_AREA * frame_area / ref_area)

    def _adaptive_block(self, blurred: np.ndarray) -> int:
        return max(11, (min(blurred.shape[:2]) // 40) | 1)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.bilateralFilter(gray, 9, 75, 75)

    def _neutralize_aruco(
        self, blurred: np.ndarray, aruco_mask: np.ndarray | None
    ) -> np.ndarray:
        if aruco_mask is None:
            return blurred
        out = blurred.copy()
        bg_pixels = out[aruco_mask == 0]
        bg = int(np.median(bg_pixels)) if bg_pixels.size else int(np.median(out))
        out[aruco_mask > 0] = bg
        return out

    def _border_pixels(
        self, blurred: np.ndarray, aruco_mask: np.ndarray | None
    ) -> np.ndarray:
        h, w = blurred.shape
        margin = max(4, min(h, w) // 25)
        strips = [
            (blurred[:margin, :], None if aruco_mask is None else aruco_mask[:margin, :]),
            (
                blurred[h - margin :, :],
                None if aruco_mask is None else aruco_mask[h - margin :, :],
            ),
            (blurred[:, :margin], None if aruco_mask is None else aruco_mask[:, :margin]),
            (
                blurred[:, w - margin :],
                None if aruco_mask is None else aruco_mask[:, w - margin :],
            ),
        ]
        pixels = []
        for strip, mask_strip in strips:
            flat = strip.ravel()
            if mask_strip is None:
                pixels.append(flat)
            else:
                kept = flat[mask_strip.ravel() == 0]
                if kept.size:
                    pixels.append(kept)
        if pixels:
            return np.concatenate(pixels)
        return blurred.ravel()

    def _is_light_background(
        self, blurred: np.ndarray, aruco_mask: np.ndarray | None
    ) -> bool:
        border = self._border_pixels(blurred, aruco_mask)
        border_med = float(np.median(border))

        if border_med >= 120:
            return True
        if border_med <= 90:
            return False

        h, w = blurred.shape
        center = blurred[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
        center_med = float(np.median(center))
        return border_med >= center_med + 10

    def _binarize_light_bg(self, blurred: np.ndarray) -> np.ndarray:
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        block = self._adaptive_block(blurred)
        adapt = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            4,
        )
        return cv2.bitwise_or(otsu, adapt)

    def _binarize_dark_bg(
        self, blurred: np.ndarray, aruco_mask: np.ndarray | None
    ) -> np.ndarray:
        block = self._adaptive_block(blurred)
        adapt = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            6,
        )

        border = self._border_pixels(blurred, aruco_mask)
        bg = float(np.percentile(border, 30))
        debris_thresh = int(min(220, bg * 1.4 + 15))
        _, relative = cv2.threshold(blurred, debris_thresh, 255, cv2.THRESH_BINARY)
        result = cv2.bitwise_and(adapt, relative)

        if np.count_nonzero(result) < 80:
            result = adapt

        return result

    def _binarize_opencv(
        self, blurred: np.ndarray, aruco_mask: np.ndarray | None
    ) -> np.ndarray:
        if self.mode == ThresholdMode.ADAPTIVE:
            block = self._adaptive_block(blurred)
            return cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block,
                4,
            )

        if self.mode == ThresholdMode.LIGHT_BG:
            _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
            return binary

        if self.mode == ThresholdMode.DARK_BG:
            _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            return binary

        if self._is_light_background(blurred, aruco_mask):
            return self._binarize_light_bg(blurred)
        return self._binarize_dark_bg(blurred, aruco_mask)

    def _pick_best_contour(
        self, contours: list, min_area: int, max_area: int
    ) -> np.ndarray | None:
        best = None
        best_score = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            score = area * (1.0 - min(circularity, 0.95))

            if score > best_score:
                best = cnt
                best_score = score

        return best

    def _contour_from_mask(
        self, frame: np.ndarray, binary: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray]:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        best = self._pick_best_contour(
            contours, self._min_area(frame), self._max_area(frame)
        )
        return best, binary

    def detect(
        self, frame: np.ndarray, aruco_mask: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.mode == ThresholdMode.MEDIAPIPE:
            binary = self._get_segmenter().create_mask(frame)
        else:
            blurred = self._neutralize_aruco(self._preprocess(frame), aruco_mask)
            binary = self._binarize_opencv(blurred, aruco_mask)

        if aruco_mask is not None:
            binary = cv2.bitwise_and(binary, cv2.bitwise_not(aruco_mask))

        return self._contour_from_mask(frame, binary)
