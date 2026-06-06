import sys

import cv2

from src.config import (
    CAMERA_BUFFER_SIZE,
    CAMERA_FOURCC,
    CAMERA_INDEX,
    CAMERA_MIN_WIDTH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)


def list_available_cameras(max_index: int = 4) -> list[dict]:
    devices = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
        best_frame = None
        for width, height in ((1920, 1080), (1280, 720), (640, 480)):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = cap.read()
            if ok and frame is not None:
                best_frame = frame
                if frame.shape[1] >= width * 0.9 and frame.shape[0] >= height * 0.9:
                    break

        if best_frame is not None:
            devices.append(
                {
                    "index": index,
                    "width": best_frame.shape[1],
                    "height": best_frame.shape[0],
                }
            )
        cap.release()
    return devices


def _try_open(index: int, backend: int | None) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None

    return cap


def _configure_resolution(cap: cv2.VideoCapture) -> tuple[int, int]:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ok, frame = cap.read()
    if ok and frame is not None:
        actual_w, actual_h = frame.shape[1], frame.shape[0]

    return actual_w, actual_h


class Camera:
    def __init__(self, index: int = CAMERA_INDEX):
        self._cap = None
        backends: list[tuple[str, int | None]] = []

        if sys.platform == "linux":
            backends = [("V4L2", cv2.CAP_V4L2), ("padrão", None)]
        else:
            backends = [("padrão", None)]

        for name, backend in backends:
            cap = _try_open(index, backend)
            if cap is not None:
                self._cap = cap
                print(f"Câmera {index} aberta (backend {name}).")
                break

        if self._cap is None:
            devices = list_available_cameras()
            msg = f"Não foi possível abrir a webcam (índice {index})."
            if devices:
                msg += "\nCâmeras disponíveis:"
                for dev in devices:
                    msg += f"\n  índice {dev['index']}: {dev['width']}x{dev['height']}px"
                msg += f"\nAltere CAMERA_INDEX em src/config.py."
            else:
                msg += "\nNenhuma câmera detectada. Verifique conexão e permissões."
            raise RuntimeError(msg)

        actual_w, actual_h = _configure_resolution(self._cap)
        print(f"Resolução de captura: {actual_w}x{actual_h}px (codec {CAMERA_FOURCC})")

        if actual_w < CAMERA_MIN_WIDTH:
            devices = list_available_cameras()
            better = [d for d in devices if d["width"] >= CAMERA_MIN_WIDTH and d["index"] != index]
            print(
                f"Aviso: resolução baixa ({actual_w}x{actual_h}). "
                "Contornos ficam imprecisos — prefira 1280x720 ou mais."
            )
            if better:
                best = max(better, key=lambda d: d["width"] * d["height"])
                alts = ", ".join(
                    f"índice {d['index']} ({d['width']}x{d['height']})"
                    for d in sorted(better, key=lambda d: d["width"] * d["height"], reverse=True)
                )
                print(f"Dica: use {alts} — altere CAMERA_INDEX em src/config.py.")
                print(f"      Recomendado: índice {best['index']} ({best['width']}x{best['height']}px).")

    def read(self):
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return False, None
        return True, frame

    def release(self):
        self._cap.release()
