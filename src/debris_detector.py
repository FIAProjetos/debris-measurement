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

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.bilateralFilter(gray, 9, 75, 75)

    def _binarize(self, blurred: np.ndarray) -> np.ndarray:
        if self.mode == ThresholdMode.ADAPTIVE:
            block = max(11, (min(blurred.shape[:2]) // 40) | 1)
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

        # AUTO: combina Otsu e adaptativo e mantém o melhor contorno candidato
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        block = max(11, (min(blurred.shape[:2]) // 40) | 1)
        adapt = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            4,
        )
        return cv2.bitwise_or(otsu, adapt)

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

            # Prefer contours with irregular shape (pedras) over noise blobs
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            score = area * (1.0 - min(circularity, 0.95))

            if score > best_score:
                best = cnt
                best_score = score

        return best

    def detect(
        self, frame: np.ndarray, aruco_mask: np.ndarray | None
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        blurred = self._preprocess(frame)
        binary = self._binarize(blurred)

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

        if aruco_mask is not None:
            binary = cv2.bitwise_and(binary, cv2.bitwise_not(aruco_mask))

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        best = self._pick_best_contour(
            contours, self._min_area(frame), self._max_area(frame)
        )

        if best is None:
            return None, binary

        return best, binary
