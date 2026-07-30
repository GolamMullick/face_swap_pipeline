"""
Runner: executes an ordered list of stages against a PipelineContext.

Treating the pipeline as data (name -> Stage instance, plus a config dict
of per-stage params) rather than hardcoded function calls means you can
reorder, disable, or swap stages (e.g. drop in a real neural face-swap
model in place of FaceReplaceStage) without touching the runner.
"""
from pipeline.context import PipelineContext
from pipeline.stages.detect_align import DetectAlignStage
from pipeline.stages.face_replace import FaceReplaceStage
from pipeline.stages.enhance import EnhanceStage
from pipeline.stages.relight import RelightStage

DEFAULT_PIPELINE = [
    DetectAlignStage(),
    FaceReplaceStage(),
    EnhanceStage(),
    RelightStage(),
]


class Pipeline:
    def __init__(self, stages=None):
        self.stages = stages if stages is not None else DEFAULT_PIPELINE

    def run(self, ctx: PipelineContext, params: dict | None = None) -> PipelineContext:
        params = params or {}
        for stage in self.stages:
            stage_params = params.get(stage.name, {})
            ctx = stage.run(ctx, **stage_params)
        return ctx

    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]
