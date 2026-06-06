from collections import deque
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from src.config import SMOOTHING_WINDOW


class MeasureStatus(Enum):
    NO_CALIBRATION = "Aguardando marcador ArUco..."
    NO_DETRITO = "Detrito não detectado"
    OK = "Medição ativa"


@dataclass
class Measurement:
    width_cm: float
    height_cm: float
    area_cm2: float
    status: MeasureStatus
    box_points: np.ndarray | None = None


class Measurer:
    def __init__(self):
        self._width_history: deque[float] = deque(maxlen=SMOOTHING_WINDOW)
        self._height_history: deque[float] = deque(maxlen=SMOOTHING_WINDOW)
        self._area_history: deque[float] = deque(maxlen=SMOOTHING_WINDOW)

    def reset(self):
        self._width_history.clear()
        self._height_history.clear()
        self._area_history.clear()

    def measure(
        self,
        contour: np.ndarray | None,
        px_per_cm: float | None,
    ) -> Measurement:
        if px_per_cm is None or px_per_cm <= 0:
            return Measurement(0, 0, 0, MeasureStatus.NO_CALIBRATION)

        if contour is None:
            return Measurement(0, 0, 0, MeasureStatus.NO_DETRITO)

        rect = cv2.minAreaRect(contour)
        w_px, h_px = rect[1]
        width_cm = min(w_px, h_px) / px_per_cm
        height_cm = max(w_px, h_px) / px_per_cm
        area_cm2 = cv2.contourArea(contour) / (px_per_cm**2)

        self._width_history.append(width_cm)
        self._height_history.append(height_cm)
        self._area_history.append(area_cm2)

        box = cv2.boxPoints(rect).astype(np.int32)
        return Measurement(
            width_cm=sum(self._width_history) / len(self._width_history),
            height_cm=sum(self._height_history) / len(self._height_history),
            area_cm2=sum(self._area_history) / len(self._area_history),
            status=MeasureStatus.OK,
            box_points=box,
        )
