"""
Every pipeline stage implements this interface. Keeping stages this
uniform is what lets the runner treat the pipeline as *data* (an ordered
list of stages + params) rather than hardcoded control flow — same idea
as a VFX studio's pipeline being configurable per-show rather than
per-shot custom code.
"""
from abc import ABC, abstractmethod
from pipeline.context import PipelineContext


class Stage(ABC):
    name: str = "unnamed_stage"

    @abstractmethod
    def run(self, ctx: PipelineContext, **params) -> PipelineContext:
        ...
