from enum import Enum

import cv2
import numpy as np

from src.config import MAX_CONTOUR_AREA, MIN_CONTOUR_AREA, MORPH_KERNEL_SIZE


class ThresholdMode(Enum):
    AUTO = "auto"
    LIGHT_BG = "claro"
    DARK_BG = "escuro"
    ADAPTIVE = "adaptativo"


class DebrisDetector:
    def __init__(self):
        self.mode = ThresholdMode.AUTO

    def close(self):
        pass

    def cycle_mode(self) -> ThresholdMode:
        modes = list(ThresholdMode)
        idx = (modes.index(self.mode) + 1) % len(modes)
        self.mode = modes[idx]
        return self.mode

    def _min_area(self, frame: np.ndarray) -> int:
        frame_area = frame.shape[0] * frame.shape[1]
        ref_area = 640 * 480
        return max(200, int(MIN_CONTOUR_AREA * frame_area / ref_area))

    def _max_area(self, frame: np.ndarray) -> int:
        frame_area = frame.shape[0] * frame.shape[1]
        ref_area = 640 * 480
        return int(MAX_CONTOUR_AREA * frame_area / ref_area)

    def _adaptive_block(self, blurred: np.ndarray, scale: int = 40) -> int:
        return max(11, (min(blurred.shape[:2]) // scale) | 1)

    def _remove_small_blobs(self, binary: np.ndarray, min_px: int = 120) -> np.ndarray:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        out = np.zeros_like(binary)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_px:
                out[labels == i] = 255
        return out

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
        block = self._adaptive_block(blurred, scale=32)
        adapt = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            10,
        )
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        result = cv2.bitwise_and(adapt, otsu)

        if np.count_nonzero(result) < 80:
            result = adapt

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=1)
        return self._remove_small_blobs(result)

    def _binarize_dark_bg(
        self, blurred: np.ndarray, aruco_mask: np.ndarray | None
    ) -> np.ndarray:
        block = self._adaptive_block(blurred, scale=32)
        adapt = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            12,
        )

        border = self._border_pixels(blurred, aruco_mask)
        bg = float(np.percentile(border, 50))
        debris_thresh = int(min(230, bg + max(40, (255 - bg) * 0.32)))
        _, relative = cv2.threshold(blurred, debris_thresh, 255, cv2.THRESH_BINARY)
        result = cv2.bitwise_and(adapt, relative)

        if np.count_nonzero(result) < 80:
            strict_thresh = int(min(240, bg + max(55, (255 - bg) * 0.42)))
            _, result = cv2.threshold(blurred, strict_thresh, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=2)
        return self._remove_small_blobs(result, min_px=150)

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

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < 0.4:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            score = area * (1.0 - min(circularity, 0.95)) * min(solidity, 1.0)

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
        blurred = self._neutralize_aruco(self._preprocess(frame), aruco_mask)
        binary = self._binarize_opencv(blurred, aruco_mask)

        if aruco_mask is not None:
            binary = cv2.bitwise_and(binary, cv2.bitwise_not(aruco_mask))

        return self._contour_from_mask(frame, binary)
