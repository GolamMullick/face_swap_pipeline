"""
Stage 3: Enhancement — three classic, well-understood filters:

  - bilateral filter   -> denoise while preserving edges (only averages
                          pixels that are both near AND similar in color)
  - CLAHE on L channel -> local contrast, tile by tile, on brightness
                          only so colors don't shift
  - unsharp mask       -> sharpen via original + k*(original - blurred)

All parameters are exposed so this stage can be re-tuned and re-run
cheaply — no face re-detection, no re-blending. This is the "artist
tweak" layer of the pipeline.
"""
import cv2
import numpy as np
from pipeline.stages.base import Stage
from pipeline.context import PipelineContext


class EnhanceStage(Stage):
    name = "enhance"

    def run(self, ctx: PipelineContext, strength: float = 1.0,
             denoise: bool = True, sharpen: bool = True, **params) -> PipelineContext:
        img = ctx.working.copy()

        if denoise:
            img = cv2.bilateralFilter(img, d=7, sigmaColor=50 * strength, sigmaSpace=50)

        # CLAHE local contrast on luminance only, to avoid color shifts
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5 * strength, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

        if sharpen:
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
            img = cv2.addWeighted(img, 1 + 0.6 * strength, blurred, -0.6 * strength, 0)

        ctx.record_pass("enhanced", img, f"denoise={denoise}, sharpen={sharpen}, strength={strength}")
        ctx.working = img
        return ctx
