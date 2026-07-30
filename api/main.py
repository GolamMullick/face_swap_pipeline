import io
import json
import os
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, FileResponse

from pipeline.context import PipelineContext
from pipeline.runner import Pipeline

app = FastAPI(
    title="Face Replacement / Enhancement / Relighting Pipeline",
    description="A configurable, stage-based image pipeline: detect+align -> face replace -> enhance -> relight.",
    version="0.1.0",
)

pipeline = Pipeline()
JOBS: dict[str, PipelineContext] = {}  # demo only — in-memory job store

_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")


def _read_image(upload_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(upload_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    return img


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(500, "Could not encode image")
    return buf.tobytes()


@app.get("/")
def root():
    """Serves the no-code web console. Machine-readable info lives at /api/info."""
    return FileResponse(os.path.join(_WEB_DIR, "index.html"))


@app.get("/api/info")
def api_info():
    return {
        "service": "face-pipeline-demo",
        "stages": pipeline.stage_names(),
        "endpoints": ["/pipeline/run", "/pipeline/{job_id}/passes", "/pipeline/{job_id}/passes/{name}"],
    }


@app.post("/pipeline/run")
async def run_pipeline(
    target: UploadFile = File(..., description="The plate — image the face will live in"),
    driver: UploadFile | None = File(None, description="Optional — source face to insert into target"),
    params: str = Form(
        "{}",
        description=(
            'JSON of per-stage params, e.g. '
            '{"enhance": {"strength": 1.2}, "relight": {"light_angle": 90, "light_strength": 0.4}}'
        ),
    ),
):
    target_img = _read_image(await target.read())
    driver_img = _read_image(await driver.read()) if driver is not None else None

    try:
        stage_params = json.loads(params) if params else {}
    except json.JSONDecodeError:
        raise HTTPException(400, "params must be valid JSON")

    ctx = PipelineContext(target=target_img, driver=driver_img)
    ctx = pipeline.run(ctx, stage_params)

    job_id = str(uuid.uuid4())
    JOBS[job_id] = ctx

    return {
        "job_id": job_id,
        "passes": list(ctx.passes.keys()),
        "log": ctx.log,
        "driver_face_detected": ctx.driver_face_box is not None,
        "target_face_detected": ctx.target_face_box is not None,
    }


@app.get("/pipeline/{job_id}/passes")
def list_passes(job_id: str):
    ctx = JOBS.get(job_id)
    if not ctx:
        raise HTTPException(404, "Job not found")
    return {"job_id": job_id, "passes": list(ctx.passes.keys()), "log": ctx.log}


@app.get("/pipeline/{job_id}/passes/{name}")
def get_pass(job_id: str, name: str):
    ctx = JOBS.get(job_id)
    if not ctx:
        raise HTTPException(404, "Job not found")
    if name not in ctx.passes:
        raise HTTPException(404, f"No such pass '{name}'. Available: {list(ctx.passes.keys())}")
    return Response(content=_encode_png(ctx.passes[name]), media_type="image/png")


@app.get("/pipeline/{job_id}/final")
def get_final(job_id: str):
    ctx = JOBS.get(job_id)
    if not ctx:
        raise HTTPException(404, "Job not found")
    return Response(content=_encode_png(ctx.working), media_type="image/png")
