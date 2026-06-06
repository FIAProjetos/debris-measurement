"""Segmentação de detrito via MediaPipe Interactive Segmenter (Tasks API)."""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python.vision.interactive_segmenter import RegionOfInterest

from src.config import SEGMENTER_INFERENCE_SCALE, SEGMENTER_MODEL_PATH


class DebrisSegmenter:
    """Segmenta o objeto sob o ponto central do frame (ideal para apontar a câmera)."""

    def __init__(self):
        if not SEGMENTER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {SEGMENTER_MODEL_PATH}\n"
                "Execute: python assets/download_segmenter_model.py"
            )

        options = vision.InteractiveSegmenterOptions(
            base_options=python.BaseOptions(model_asset_path=str(SEGMENTER_MODEL_PATH)),
            output_category_mask=True,
        )
        self._segmenter = vision.InteractiveSegmenter.create_from_options(options)

    def close(self):
        self._segmenter.close()

    def create_mask(
        self,
        frame: np.ndarray,
        roi_x: float = 0.5,
        roi_y: float = 0.5,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = SEGMENTER_INFERENCE_SCALE
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        roi = RegionOfInterest(
            format=RegionOfInterest.Format.KEYPOINT,
            keypoint=containers.keypoint.NormalizedKeypoint(x=roi_x, y=roi_y),
        )
        result = self._segmenter.segment(mp_image, roi)

        mask = result.category_mask.numpy_view().squeeze().astype(np.uint8)
        mask = (mask > 0).astype(np.uint8) * 255
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
