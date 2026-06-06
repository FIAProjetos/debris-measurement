import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarksConnections, drawing_utils

from src.config import (
    GESTURE_COOLDOWN_FRAMES,
    GESTURE_INFERENCE_SCALE,
    GESTURE_PROCESS_EVERY_N_FRAMES,
    HAND_DETECTION_CONFIDENCE,
    HAND_MODEL_PATH,
    HAND_TRACKING_CONFIDENCE,
)


class GestureController:
    """Detecta gesto de captura via MediaPipe HandLandmarker (Tasks API)."""

    def __init__(self):
        if not HAND_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {HAND_MODEL_PATH}\n"
                "Execute: python assets/download_hand_model.py"
            )

        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=HAND_TRACKING_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0
        self._cooldown = 0
        self._prev_fist = False
        self._frame_counter = 0

    def close(self):
        self._landmarker.close()

    def _is_fist(self, landmarks) -> bool:
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        folded = all(
            landmarks[tip].y > landmarks[pip].y for tip, pip in zip(tips, pips)
        )
        thumb_closed = landmarks[4].x > landmarks[3].x
        return folded and thumb_closed

    def _is_pointing(self, landmarks) -> bool:
        index_up = landmarks[8].y < landmarks[6].y
        others_down = all(
            landmarks[tip].y > landmarks[pip].y
            for tip, pip in [(12, 10), (16, 14), (20, 18)]
        )
        return index_up and others_down

    def process(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """Retorna (trigger_save, frame_anotado)."""
        if self._cooldown > 0:
            self._cooldown -= 1

        self._frame_counter += 1
        if self._frame_counter % GESTURE_PROCESS_EVERY_N_FRAMES != 0:
            return False, frame

        h, w = frame.shape[:2]
        scale = GESTURE_INFERENCE_SCALE
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
        self._timestamp_ms += 33

        trigger = False
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            drawing_utils.draw_landmarks(
                frame,
                hand,
                HandLandmarksConnections.HAND_CONNECTIONS,
            )

            is_fist = self._is_fist(hand)
            is_pointing = self._is_pointing(hand)

            if self._prev_fist and is_pointing and self._cooldown == 0:
                trigger = True
                self._cooldown = GESTURE_COOLDOWN_FRAMES

            self._prev_fist = is_fist

        return trigger, frame
