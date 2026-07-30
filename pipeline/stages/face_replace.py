"""
Stage 2: Face replacement.

Pipeline within the stage:
  1. Build a soft-edged elliptical mask (faces are oval; the Gaussian
     blur feathers the boundary so the blend fades instead of stopping).
  2. Color-match the driver face to the target region (Reinhard-style
     transfer in LAB space) BEFORE blending — this matters: the Poisson
     blend smooths color differences at the seam, and if the color gap
     is large that smoothing washes the driver's features out toward the
     target's colors. Matching first means the blend has almost nothing
     to correct, so the driver's identity survives.
  3. Poisson-blend with cv2.seamlessClone using NORMAL_CLONE — not
     MIXED_CLONE. MIXED_CLONE keeps whichever gradients (source's or
     destination's) are stronger at each pixel, which for a face swap
     can silently keep the ORIGINAL target's features if they're
     higher-contrast than the driver's. NORMAL_CLONE fully transfers
     the driver's appearance and only blends at the mask edge, which is
     what identity replacement needs.
  4. If seamlessClone throws (it can, when the face box touches the
     image border), fall back to a feathered alpha blend.

If no driver face was detected, this stage is a passthrough — the
pipeline still runs enhancement/relighting on the target alone.
"""
import cv2
import numpy as np
from pipeline.stages.base import Stage
from pipeline.context import PipelineContext


class FaceReplaceStage(Stage):
    name = "face_replace"

    def run(self, ctx: PipelineContext, color_match: bool = True, **params) -> PipelineContext:
        if ctx.driver_face_box is None or ctx.target_face_box is None:
            ctx.record_pass("raw_swap", ctx.target, "no driver face detected — passthrough")
            ctx.working = ctx.target
            return ctx

        tx, ty, tw, th = ctx.target_face_box
        driver_face = ctx.working  # aligned crop from stage 1
        driver_resized = cv2.resize(driver_face, (tw, th), interpolation=cv2.INTER_LANCZOS4)

        mask = np.zeros((th, tw), dtype=np.uint8)
        cv2.ellipse(mask, (tw // 2, th // 2), (int(tw * 0.42), int(th * 0.48)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)

        if color_match:
            target_region = ctx.target[ty:ty + th, tx:tx + tw]
            driver_resized = _match_color_lab(driver_resized, target_region, mask)
            ctx.log.append("[face_replace] color-matched driver face to target region (LAB)")

        swapped = self._blend(ctx.target, driver_resized, mask, tx, ty, tw, th)

        ctx.record_pass("raw_swap", swapped, "Poisson-blended driver face onto target face box")
        ctx.working = swapped
        return ctx

    def _blend(self, target, driver_resized, mask, tx, ty, tw, th) -> np.ndarray:
        base = target.copy()  # never mutate the caller's array
        center = (tx + tw // 2, ty + th // 2)
        try:
            return cv2.seamlessClone(driver_resized, base, mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            # seamlessClone can fail if the box touches the image border; fall back to alpha blend
            swapped = base.copy()
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            region = swapped[ty:ty + th, tx:tx + tw].astype(np.float32)
            blended = region * (1 - alpha) + driver_resized.astype(np.float32) * alpha
            swapped[ty:ty + th, tx:tx + tw] = np.clip(blended, 0, 255).astype(np.uint8)
            return swapped


def _match_color_lab(source: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reinhard color transfer: shift `source`'s LAB mean/std (inside `mask`)
    to match `reference`'s, so a face shot under different lighting/camera
    settings lands in the target's color world before compositing."""
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)

    m = mask.astype(bool)
    if m.sum() < 16:  # degenerate mask — skip rather than divide by noise
        return source

    out = src_lab.copy()
    for c in range(3):
        s_vals, r_vals = src_lab[..., c][m], ref_lab[..., c][m]
        s_mean, s_std = s_vals.mean(), s_vals.std()
        r_mean, r_std = r_vals.mean(), r_vals.std()
        if s_std < 1e-3:
            continue
        out[..., c] = (src_lab[..., c] - s_mean) * (r_std / s_std) + r_mean

    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
