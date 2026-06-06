"""Baixa o modelo HandLandmarker exigido pelo MediaPipe Tasks API."""

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
OUT_PATH = Path(__file__).parent / "hand_landmarker.task"


def main():
    if OUT_PATH.exists():
        print(f"Modelo já existe: {OUT_PATH}")
        return

    print(f"Baixando modelo para {OUT_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, OUT_PATH)
    print("Download concluído.")


if __name__ == "__main__":
    main()
