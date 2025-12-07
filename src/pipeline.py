"""
High-level pipeline stub for ability VFX detection and future fusion with minimap/HUD cues.
"""
from pathlib import Path
from typing import Any

from ultralytics import YOLO


class VFXPipeline:
    def __init__(self, weights: Path, conf: float = 0.25, iou: float = 0.45) -> None:
        self.model = YOLO(str(weights))
        self.conf = conf
        self.iou = iou

    def predict(self, source: str | Path, save_dir: Path | None = None) -> Any:
        kwargs: dict[str, Any] = {"conf": self.conf, "iou": self.iou}
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            kwargs.update({"project": str(save_dir), "name": "preds", "save": True, "exist_ok": True})
        return self.model.predict(source=source, **kwargs)
