# Face Pipeline Studio

A face replacement, enhancement, and relighting pipeline with a web UI.
Upload two photos, and it detects the faces with a neural detector,
swaps one onto the other with color-matched Poisson blending, cleans up
the result, and relights it to match the scene — with every intermediate
step saved as an inspectable "pass."

Runs entirely on CPU. No GPU, no model downloads, no API keys.

```
detect + align  →  face replace  →  enhance  →  relight
```

## Quick start

**With Docker:**

```bash
docker build -t face-pipeline .
docker run -p 8001:8001 face-pipeline
```

**With Python (3.11+):**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8001
```

Then open **http://localhost:8001** — drag in a target photo (and
optionally a driver face), click **Run pipeline**, and use the wipe
slider to compare before/after. API docs are at `/docs`.

There's also a Streamlit version of the UI (runs the pipeline
in-process, no server):

```bash
python -m venv .venv-streamlit && source .venv-streamlit/bin/activate
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

## How it works

Each stage receives a shared `PipelineContext`, transforms the working
image, and records a named, frozen copy of its output (a *pass*).
Nothing is destructive — if the final result looks wrong, every
intermediate step is still there to inspect.

**1. `detect_align`** — face detection with **YuNet**
(`cv2.FaceDetectorYN`), a small neural detector whose ONNX weights
(~310KB) are bundled in `pipeline/models/`. It returns a box plus five
facial landmarks; the measured eye positions are used to rotate the
driver's face level before compositing. If the model file is missing,
the stage falls back to OpenCV's classic Haar cascade and logs which
detector ran. (In testing, Haar found nothing at 30° head rotation and
returned a false positive at 40°, where YuNet stayed correct — hence
the upgrade.)

**2. `face_replace`** — the driver face is resized to the target's face
box, **color-matched** to the target region first (Reinhard transfer in
LAB space — shifting its color statistics into the target photo's
"color world"), then composited with `cv2.seamlessClone` using
`NORMAL_CLONE` inside a feathered elliptical mask. Two details matter:

- *Color matching happens before the blend*, because Poisson blending
  smooths color differences at the seam — a large color gap gets
  "corrected" by washing the driver's features out toward the target's
  colors. Matching first leaves the blend almost nothing to correct.
- *`NORMAL_CLONE`, not `MIXED_CLONE`.* MIXED_CLONE keeps whichever
  gradients are stronger at each pixel, which for a face swap can
  silently preserve the original target's features. This was a real
  bug: measured pixel change inside the face box went from 10.1
  (MIXED) → 15.8 (NORMAL) → 18.1 (NORMAL + color matching).

If `seamlessClone` throws (face box touching the image border), a
feathered alpha blend is the fallback.

**3. `enhance`** — bilateral filtering (edge-preserving denoise), CLAHE
on the L channel only (local contrast without color shift), and an
unsharp mask. All parameters exposed; re-runs in milliseconds without
re-detecting or re-blending.

**4. `relight`** — estimates the scene's ambient lighting by heavily
blurring the target's luminance channel, and nudges the composited face
toward it; plus an adjustable synthetic directional light (angle +
strength) applied inside the face mask. This is the cheap
"downstream tweak" stage by design.

## Design notes

- **Pipeline as data.** The pipeline is an ordered list of `Stage`
  objects (`pipeline/runner.py`); every stage implements
  `run(ctx, **params) -> ctx`. Reordering, removing, or swapping a
  stage — including replacing the classic blend with a learned model —
  means editing a list, not rewriting control flow.
- **Non-destructive passes.** Every stage snapshot is retrievable via
  `GET /pipeline/{job_id}/passes/{name}` (`detection`, `raw_swap`,
  `enhanced`, `relit`).
- **Fallbacks are logged.** Neural detector → Haar; seamlessClone →
  alpha blend. Silent fallbacks hide problems; logged ones surface them
  in the UI's stage log.
- **Known shortcut:** job results live in an in-memory dict, so they
  don't survive a restart and don't scale past one process. Redis or
  SQLite is the obvious next step.

## API

```bash
curl -X POST \
  -F "target=@plate.jpg" \
  -F "driver=@face.jpg" \
  -F 'params={"enhance": {"strength": 1.2}, "relight": {"light_angle": 90}}' \
  http://127.0.0.1:8001/pipeline/run
# -> {"job_id": "...", "passes": ["detection","raw_swap","enhanced","relit"], "log": [...]}

curl http://127.0.0.1:8001/pipeline/<job_id>/passes/relit -o result.png
```

| Endpoint | What it does |
|---|---|
| `POST /pipeline/run` | Run the full pipeline on uploaded image(s) |
| `GET /pipeline/{id}/passes` | List pass names + per-stage log |
| `GET /pipeline/{id}/passes/{name}` | One pass as PNG |
| `GET /pipeline/{id}/final` | Final image as PNG |

Per-stage parameters (all optional): `face_replace.color_match` (bool),
`enhance.strength/denoise/sharpen`, `relight.match_plate_lighting/
light_angle/light_strength`.

## Project layout

```
web/index.html               web UI (talks to the API over HTTP)
streamlit_app.py             alternative UI (imports the pipeline in-process)
api/main.py                  FastAPI server
pipeline/
  context.py                 shared state object passed through stages
  runner.py                  executes the stage list
  stages/                    one file per stage
  models/                    bundled YuNet face-detection ONNX
Dockerfile, docker-compose.yml
```

## Troubleshooting

- **"No face detected"** — the detector needs a reasonably clear face
  in each photo; very small, heavily occluded, or extreme-profile faces
  may not be found. The stage log says exactly which image failed.
- **`ImportError: libGL.so.1` (Docker/Linux)** — OpenCV needs a few
  system libraries even in its headless build; the provided Dockerfile
  installs them (`libgl1`, `libglib2.0-0`, …).
- **`cv2` has no attribute errors** — usually a local file/folder named
  `cv2` shadowing the package, or mixed OpenCV variants installed.
  Check `python -c "import cv2; print(cv2.__file__)"`.

## Credits

- Face detection: [YuNet](https://github.com/ShiqiYu/libfacedetection.train)
  by Shiqi Yu (BSD 3-Clause) — ONNX weights bundled in
  `pipeline/models/`.
- Built with OpenCV, FastAPI, and Streamlit.

## License

MIT — see [LICENSE](LICENSE).
