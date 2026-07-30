# Face Replacement / Enhancement / Relighting Pipeline
#
# Build:  docker build -t face-pipeline .
# Run:    docker run -p 8001:8001 face-pipeline
# Then:   open http://localhost:8001

FROM python:3.12-slim

# opencv-python-headless still dynamically links against a few system
# libs (libGL, glib, X render) even in "headless" builds — without these
# you'll hit `ImportError: libGL.so.1: cannot open shared object file`
# on `import cv2`.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app
COPY api ./api
COPY pipeline ./pipeline
COPY web ./web

# Run as a non-root user
RUN useradd -m appuser
USER appuser

ENV PYTHONUNBUFFERED=1

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/api/info', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
