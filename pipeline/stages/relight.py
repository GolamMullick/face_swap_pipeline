"""
Stage 4: Relighting.

Two real, classic techniques combined:

1. Luminance-field transfer — constantly changing lighting conditions
   are one of the hardest parts of compositing work.
   Here we estimate the plate's local lighting field (a heavily blurred
   luminance map of the area around the face) and nudge the composited
   face's luminance toward it, per-pixel, via a mean-normalized transfer.
   This is the same idea as matching a comp element to a plate's key light
   before doing anything fancier.

2. Synthetic directional light — an adjustable "virtual light" (angle +
   strength) added only inside the face mask, so an artist could push a
   rim light or counter the plate's ambient light without re-running any
   inference. Mirrors the talk's "exposure tweak" pattern.
"""
import cv2
import numpy as np
from pipeline.stages.base import Stage
from pipeline.context import PipelineContext


def _face_mask(shape, box, feather=25):
    mask = np.zeros(shape[:2], dtype=np.float32)
    if box is None:
        return mask
    x, y, w, h = box
    cv2.ellipse(mask, (x + w // 2, y + h // 2), (int(w * 0.45), int(h * 0.52)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather)
    return mask


def _directional_light(shape, angle_deg: float, strength: float) -> np.ndarray:
    """Simple linear light gradient plane, angle measured from +x axis."""
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx / w) - 0.5
    yy = (yy / h) - 0.5
    theta = np.radians(angle_deg)
    proj = xx * np.cos(theta) + yy * np.sin(theta)   # -0.5..0.5 along light direction
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    return (proj * 2 - 1) * strength * 60.0            # scaled to ~L-channel units


class RelightStage(Stage):
    name = "relight"

    def run(self, ctx: PipelineContext, light_angle: float = 45.0,
             light_strength: float = 0.3, match_plate_lighting: bool = True,
             **params) -> PipelineContext:
        img = ctx.working.copy()
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l, a, b = cv2.split(lab)

        mask = _face_mask(img.shape, ctx.target_face_box)

        if match_plate_lighting and ctx.target_face_box is not None:
            # Estimate the plate's ambient lighting field via heavy blur,
            # then pull the face region's luminance toward its local mean.
            plate_field = cv2.GaussianBlur(l, (0, 0), sigmaX=41)
            x, y, w, h = ctx.target_face_box
            face_mean = l[y:y + h, x:x + w].mean()
            target_mean = plate_field[y:y + h, x:x + w].mean()
            delta = (target_mean - face_mean)
            l = l + delta * mask

        if light_strength > 0:
            light = _directional_light(img.shape, light_angle, light_strength)
            l = l + light * mask

        l = np.clip(l, 0, 255)
        out = cv2.cvtColor(cv2.merge([l, a, b]).astype(np.uint8), cv2.COLOR_LAB2BGR)

        ctx.record_pass(
            "relit", out,
            f"match_plate_lighting={match_plate_lighting}, light_angle={light_angle}, light_strength={light_strength}",
        )
        ctx.working = out
        return ctx
