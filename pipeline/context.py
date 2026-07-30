"""
PipelineContext: the single object passed through every stage.

Built on the principle that pipeline outputs shouldn't be a single
flat/baked image — every stage writes its output into `passes` under its
own name, so nothing downstream is destructive and every intermediate
result stays inspectable (and swappable) on its own.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PipelineContext:
    target: np.ndarray                     # the "plate" - BGR image the face will live in
    driver: np.ndarray | None = None        # optional source face to insert into target

    target_face_box: tuple | None = None    # (x, y, w, h) in target
    driver_face_box: tuple | None = None    # (x, y, w, h) in driver

    working: np.ndarray | None = None       # current image as it flows through stages

    passes: dict = field(default_factory=dict)   # name -> BGR np.ndarray, one per stage
    log: list = field(default_factory=list)       # human-readable stage notes, for debugging/demo

    def record_pass(self, name: str, image: np.ndarray, note: str = ""):
        self.passes[name] = image.copy()
        if note:
            self.log.append(f"[{name}] {note}")
