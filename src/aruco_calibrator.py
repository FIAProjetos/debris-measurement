from collections import deque

import cv2
import numpy as np

from src.config import ARUCO_DICT, MARKER_ID, MARKER_SIZE_CM, SMOOTHING_WINDOW


class ArucoCalibrator:
    def __init__(self):
        self._dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
        self._params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)
        self._px_per_cm_history: deque[float] = deque(maxlen=SMOOTHING_WINDOW)

    def reset(self):
        self._px_per_cm_history.clear()

    @property
    def is_calibrated(self) -> bool:
        return len(self._px_per_cm_history) > 0

    @property
    def px_per_cm(self) -> float | None:
        if not self._px_per_cm_history:
            return None
        return sum(self._px_per_cm_history) / len(self._px_per_cm_history)

    def process(self, frame: np.ndarray) -> tuple[list, np.ndarray | None]:
        """Detecta ArUco e retorna (corners, mask) ou ([], None)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None or MARKER_ID not in ids.flatten():
            return [], None

        idx = int(np.where(ids.flatten() == MARKER_ID)[0][0])
        marker_corners = [corners[idx]]

        pts = marker_corners[0][0]
        side_px = (
            np.linalg.norm(pts[0] - pts[1])
            + np.linalg.norm(pts[1] - pts[2])
            + np.linalg.norm(pts[2] - pts[3])
            + np.linalg.norm(pts[3] - pts[0])
        ) / 4.0
        px_per_cm = side_px / MARKER_SIZE_CM
        self._px_per_cm_history.append(px_per_cm)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, pts.astype(np.int32), 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=2)

        return marker_corners, mask

    def draw(self, frame: np.ndarray, marker_corners: list) -> np.ndarray:
        if marker_corners:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, np.array([[MARKER_ID]]))
        return frame
