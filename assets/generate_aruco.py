"""Gera marcador ArUco para impressão em escala real."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ARUCO_DICT, MARKER_ID, MARKER_SIZE_CM  # noqa: E402

# 300 DPI → pixels por cm
DPI = 300
PX_PER_CM = DPI / 2.54
MARKER_PX = int(MARKER_SIZE_CM * PX_PER_CM)
BORDER_PX = int(PX_PER_CM)  # borda branca de 1 cm


def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, MARKER_ID, MARKER_PX)

    canvas = np.ones(
        (marker_img.shape[0] + 2 * BORDER_PX, marker_img.shape[1] + 2 * BORDER_PX),
        dtype=np.uint8,
    ) * 255
    canvas[BORDER_PX:-BORDER_PX, BORDER_PX:-BORDER_PX] = marker_img

    out_path = Path(__file__).parent / f"aruco_marker_id{MARKER_ID}.png"
    cv2.imwrite(str(out_path), canvas)
    print(f"Marcador salvo em: {out_path}")
    print(f"Tamanho do quadrado preto: {MARKER_SIZE_CM} cm x {MARKER_SIZE_CM} cm")
    print("Imprima em escala 100% (tamanho real), sem redimensionar.")


if __name__ == "__main__":
    main()
