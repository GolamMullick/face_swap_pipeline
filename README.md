# Face Pipeline Studio

Swap a face from one photo into another — then clean it up and fix the
lighting so it looks like it belongs there.

You upload two photos in a web page, click one button, and get the
result. You can also see every in-between step, and compare before and
after with a slider.

```
find the face  →  swap it in  →  clean it up  →  fix the lighting
```

Works on any computer. No GPU needed, nothing to sign up for.

---

## How to run it

You need **one** of these installed:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (easiest), or
- [Python 3.11+](https://www.python.org/downloads/)

### With Docker

Open a terminal in this folder and run:

```bash
docker build -t face-pipeline .
docker run -p 8001:8001 face-pipeline
```

Then open **http://localhost:8001** in your browser.

### With Python

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8001
```

Then open **http://localhost:8001** in your browser.

---

## How to use it

1. Drag a photo into the **Target** box — the photo the face will go into.
2. Drag another photo into the **Driver** box — the face you want to use.
   (You can skip this one — then it just cleans up and relights the
   target photo.)
3. Click **Run pipeline**.
4. Drag the blue handle on the result to compare before and after.
5. The four small thumbnails underneath are the in-between steps —
   click one to view it big.
6. Try the sliders on the right and run it again to see what changes.

**Tip:** use clear, front-facing photos. If a face can't be found, the
"Stage log" panel tells you which photo was the problem.

---

## What each step does

1. **Find the face** — a small neural network (included in this repo,
   no download needed) finds the face in each photo, plus the eyes,
   nose, and mouth. If the eyes aren't level, the face gets straightened.
2. **Swap it in** — the new face is first color-adjusted to match the
   target photo's lighting and skin tone, then blended in with a
   technique that hides the seam (the same kind of math photo editors
   use).
3. **Clean it up** — reduces noise, improves contrast, and sharpens.
4. **Fix the lighting** — nudges the face's brightness to match the
   scene around it, and lets you add your own light from any direction
   with the dial.

Every step saves its own image, so nothing is ever lost — you can
always look at any step on its own.

---


---

## Common problems

- **"No face detected"** — use a clearer, more front-facing photo.
  Sunglasses, side profiles, and tiny faces are hard to detect.
- **`docker: command not found`** — Docker Desktop isn't installed or
  isn't running yet.
- **Port 8001 already in use** — run with a different port:
  `docker run -p 8002:8001 face-pipeline`, then open
  http://localhost:8002.
- **An error mentioning `libGL.so.1` (Linux)** — use the Docker option;
  the Dockerfile installs the system libraries this needs.

---

## For developers

You can also call it without the web page:

```bash
curl -X POST \
  -F "target=@photo1.jpg" \
  -F "driver=@photo2.jpg" \
  http://127.0.0.1:8001/pipeline/run
```

Interactive API docs are at http://localhost:8001/docs once it's
running. The pipeline itself is just an ordered list of steps
(`pipeline/runner.py`) — each step takes an image in and passes an
image on, so steps are easy to add, remove, or replace.

---

