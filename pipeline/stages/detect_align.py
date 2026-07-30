"""
Stage 1: Detect + align — upgraded.

Primary detector: YuNet (cv2.FaceDetectorYN), a small modern neural face
detector bundled with the project (pipeline/models/yunet_n_320_320.onnx,
~310KB, from the original author's repo). Compared to the old Haar
cascade it is far more robust to pose, lighting, and partial occlusion,
and it returns five facial landmarks (eyes, nose tip, mouth corners) —
which lets alignment use the *actual measured eye positions* instead of
a second, flakier Haar eye-cascade pass.

Fallback: if the ONNX file is missing or FaceDetectorYN isn't available
in the installed OpenCV, we quietly fall back to the original Haar
cascade path and log which detector was used — same graceful-degradation
pattern used throughout this project.
"""
import os
import cv2
import numpy as np
from pipeline.stages.base import Stage
from pipeline.context import PipelineContext

_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "yunet_n_320_320.onnx")

_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

_yunet = None
if hasattr(cv2, "FaceDetectorYN") and os.path.exists(_MODEL_PATH):
    try:
        _yunet = cv2.FaceDetectorYN.create(_MODEL_PATH, "", (320, 320), score_threshold=0.6)
    except cv2.error:
        _yunet = None


def _detect_yunet(img: np.ndarray):
    """Returns (box, landmarks) for the highest-confidence face, or (None, None).
    landmarks is a (5, 2) float array: right eye, left eye, nose, mouth corners."""
    h, w = img.shape[:2]
    _yunet.setInputSize((w, h))
    _, faces = _yunet.detect(img)
    if faces is None or len(faces) == 0:
        return None, None
    best = max(faces, key=lambda f: f[14])
    x, y, bw, bh = best[:4].astype(int)
    # clamp to image bounds — YuNet boxes can slightly overshoot edges
    x, y = max(0, x), max(0, y)
    bw, bh = min(bw, w - x), min(bh, h - y)
    landmarks = best[4:14].reshape(5, 2)
    return (x, y, bw, bh), landmarks


def _detect_haar(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None, None
    return tuple(max(faces, key=lambda f: f[2] * f[3])), None


def _detect(img: np.ndarray):
    """Returns (box, landmarks, detector_name)."""
    if _yunet is not None:
        box, lms = _detect_yunet(img)
        if box is not None:
            return box, lms, "yunet"
    box, lms = _detect_haar(img)
    return box, lms, "haar"


def _eye_angle_from_landmarks(landmarks: np.ndarray) -> float:
    """Rotation (degrees) to level the eyes, from YuNet's measured eye points."""
    right_eye, left_eye = landmarks[0], landmarks[1]
    dy = left_eye[1] - right_eye[1]
    dx = left_eye[0] - right_eye[0]
    return float(np.degrees(np.arctan2(dy, dx)))


def _eye_angle_haar(face_img: np.ndarray) -> float:
    """Legacy fallback: estimate eye angle with a Haar eye cascade."""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    eyes = _eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)
    if len(eyes) < 2:
        return 0.0
    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes
    c1 = (x1 + w1 / 2, y1 + h1 / 2)
    c2 = (x2 + w2 / 2, y2 + h2 / 2)
    return float(np.degrees(np.arctan2(c2[1] - c1[1], c2[0] - c1[0])))


class DetectAlignStage(Stage):
    name = "detect_align"

    def run(self, ctx: PipelineContext, **params) -> PipelineContext:
        target_box, _, target_detector = _detect(ctx.target)
        ctx.target_face_box = target_box

        preview = ctx.target.copy()
        if target_box:
            x, y, w, h = target_box
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        ctx.record_pass("detection", preview, f"target face box: {target_box} (detector: {target_detector})")

        if ctx.driver is not None:
            driver_box, driver_lms, driver_detector = _detect(ctx.driver)
            ctx.driver_face_box = driver_box
            if driver_box:
                x, y, w, h = driver_box
                crop = ctx.driver[y:y + h, x:x + w]

                if driver_lms is not None:
                    angle = _eye_angle_from_landmarks(driver_lms)
                else:
                    angle = _eye_angle_haar(crop)

                if abs(angle) > 1.0:
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                    crop = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                ctx.working = crop  # aligned driver face crop, ready for stage 2
                ctx.log.append(
                    f"[align] driver face box: {driver_box} (detector: {driver_detector}), "
                    f"eye-angle correction: {angle:.1f} deg"
                )

        ctx.working = ctx.working if ctx.working is not None else ctx.target
        return ctx
