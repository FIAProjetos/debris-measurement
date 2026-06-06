import cv2
import numpy as np

from src.debris_detector import ThresholdMode
from src.measurer import MeasureStatus, Measurement


class Renderer:
    def draw(
        self,
        frame: np.ndarray,
        measurement: Measurement,
        contour: np.ndarray | None,
        threshold_mode: ThresholdMode,
        debug_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        out = frame.copy()

        if threshold_mode == ThresholdMode.MEDIAPIPE:
            self._draw_crosshair(out)

        if debug_mask is not None:
            small = cv2.resize(debug_mask, (out.shape[1] // 4, out.shape[0] // 4))
            colored = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
            h, w = colored.shape[:2]
            out[10 : 10 + h, 10 : 10 + w] = colored
            cv2.rectangle(out, (10, 10), (10 + w, 10 + h), (255, 255, 255), 1)
            cv2.putText(
                out,
                "mascara (D)",
                (10, 10 + h + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

        if contour is not None:
            cv2.drawContours(out, [contour], -1, (0, 255, 0), 2)

        if measurement.box_points is not None:
            cv2.drawContours(out, [measurement.box_points], 0, (255, 200, 0), 2)

        self._draw_status_bar(out, measurement, threshold_mode)
        return out

    def _draw_crosshair(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        size = 20
        color = (0, 255, 255)
        cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1)
        cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1)

    def _draw_status_bar(
        self,
        frame: np.ndarray,
        measurement: Measurement,
        threshold_mode: ThresholdMode,
    ):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        if measurement.status == MeasureStatus.OK:
            color = (0, 255, 0)
            lines = [
                f"Largura: {measurement.width_cm:.2f} cm",
                f"Altura:  {measurement.height_cm:.2f} cm",
                f"Area:    {measurement.area_cm2:.2f} cm2",
            ]
        else:
            color = (0, 165, 255)
            lines = [measurement.status.value]

        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (10, 28 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        help_text = f"Modo: {threshold_mode.value} | S=salvar T=modo D=mascara R=reset Q=sair"
        cv2.putText(
            frame,
            help_text,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
