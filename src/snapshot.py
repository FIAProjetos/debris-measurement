import json
from datetime import datetime
from pathlib import Path

import cv2

from src.config import OUTPUT_DIR
from src.measurer import Measurement


def save_snapshot(frame, measurement: Measurement, px_per_cm: float | None) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = OUTPUT_DIR / f"detrito_{timestamp}"

    png_path = base.with_suffix(".png")
    json_path = base.with_suffix(".json")

    cv2.imwrite(str(png_path), frame)

    metadata = {
        "timestamp": timestamp,
        "width_cm": round(float(measurement.width_cm), 3),
        "height_cm": round(float(measurement.height_cm), 3),
        "area_cm2": round(float(measurement.area_cm2), 3),
        "px_per_cm": round(float(px_per_cm), 3) if px_per_cm is not None else None,
        "status": measurement.status.value,
    }
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Imagem salva: {png_path}")
    print(f"Metadados:    {json_path}")
    return png_path
