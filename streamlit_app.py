"""
Streamlit UI for the face pipeline.

Unlike web/index.html (which talks to the FastAPI server over HTTP),
this app imports the pipeline package directly and runs it in-process —
one command, no separate server to keep running.

Run it with:
    streamlit run streamlit_app.py
"""
import json

import cv2
import numpy as np
import streamlit as st

from pipeline.context import PipelineContext
from pipeline.runner import Pipeline

st.set_page_config(page_title="Face Pipeline", page_icon="🎬", layout="wide")

PASS_ORDER = ["detection", "raw_swap", "enhanced", "relit"]
PASS_LABELS = {
    "detection": "1 · Detected face",
    "raw_swap": "2 · Face replaced",
    "enhanced": "3 · Enhanced",
    "relit": "4 · Relit (final)",
}


@st.cache_resource
def get_pipeline() -> Pipeline:
    return Pipeline()


def read_upload(uploaded_file) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def make_compare_image(before: np.ndarray, after: np.ndarray, split_pct: int) -> np.ndarray:
    """Builds a single wipe image: `before` on the left, `after` on the right,
    split at split_pct, with a thin line marking the boundary — this is
    what makes the slider below feel like a real before/after compare
    without needing any JavaScript."""
    h, w = before.shape[:2]
    after_resized = cv2.resize(after, (w, h)) if after.shape[:2] != (h, w) else after
    combined = before.copy()
    split_x = int(w * split_pct / 100)
    combined[:, split_x:] = after_resized[:, split_x:]
    if 0 < split_x < w:
        combined[:, max(0, split_x - 1):split_x + 1] = (79, 209, 232)  # rim-light cyan, BGR
    return combined


# ---------------------------------------------------------------------------
# Sidebar — controls, wrapped in a form so dragging a slider doesn't
# re-run the whole pipeline on every tick, only on "Run pipeline"
# ---------------------------------------------------------------------------

st.sidebar.title("🎬 Face Pipeline")
st.sidebar.caption("detect + align → face replace → enhance → relight")

with st.sidebar.form("controls"):
    st.subheader("Images")
    target_upload = st.file_uploader("Target (the plate)", type=["png", "jpg", "jpeg"])
    driver_upload = st.file_uploader("Driver (face to insert) — optional", type=["png", "jpg", "jpeg"])

    st.subheader("Face replace")
    color_match = st.checkbox("Color-match driver to target", value=True)

    st.subheader("Enhance")
    strength = st.slider("Strength", 0.1, 3.0, 1.0, 0.1)
    col1, col2 = st.columns(2)
    denoise = col1.checkbox("Denoise", value=True)
    sharpen = col2.checkbox("Sharpen", value=True)

    st.subheader("Relight")
    match_plate = st.checkbox("Match plate lighting", value=True)
    light_strength = st.slider("Light strength", 0.0, 1.0, 0.3, 0.05)
    light_angle = st.slider("Light angle (°)", 0, 360, 45)

    submitted = st.form_submit_button("▶ Run pipeline", use_container_width=True, type="primary")

st.sidebar.caption(
    "All processing runs locally with OpenCV — no GPU needed."
)

# ---------------------------------------------------------------------------
# Run the pipeline on submit, cache results in session_state
# ---------------------------------------------------------------------------

if submitted:
    target_img = read_upload(target_upload)
    driver_img = read_upload(driver_upload)

    if target_img is None:
        st.error("Add a target image first — that's the shot the face will live in.")
    else:
        params = {
            "face_replace": {"color_match": color_match},
            "enhance": {
                "strength": strength,
                "denoise": denoise,
                "sharpen": sharpen,
            },
            "relight": {
                "match_plate_lighting": match_plate,
                "light_angle": light_angle,
                "light_strength": light_strength,
            },
        }
        with st.spinner("Running pipeline…"):
            ctx = PipelineContext(target=target_img, driver=driver_img)
            ctx = get_pipeline().run(ctx, params)

        st.session_state["passes"] = {k: ctx.passes[k] for k in PASS_ORDER if k in ctx.passes}
        st.session_state["log"] = ctx.log
        st.session_state["target_face_detected"] = ctx.target_face_box is not None
        st.session_state["driver_face_detected"] = ctx.driver_face_box is not None
        st.session_state["had_driver"] = driver_img is not None
        st.session_state["original"] = target_img

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if "passes" not in st.session_state:
    st.info("👈 Add a target photo in the sidebar and click **Run pipeline** to get started.")
    st.caption(
        "No test photo handy? `pip install scikit-image` then:\n\n"
        "```python\nimport cv2, skimage.data as d\n"
        "cv2.imwrite('target.png', cv2.cvtColor(d.astronaut(), cv2.COLOR_RGB2BGR))\n```"
    )
else:
    if not st.session_state["target_face_detected"]:
        st.warning("No face found in the target image — enhance/relight still ran on the plate as-is.")
    elif st.session_state["had_driver"] and not st.session_state["driver_face_detected"]:
        st.warning("Target face found, but no face detected in the driver image — swap was skipped.")
    else:
        st.success(f"Done — {len(st.session_state['passes'])} passes generated.")

    passes = st.session_state["passes"]
    original = st.session_state["original"]

    st.subheader("Before / after")
    if "relit" in passes:
        split = st.slider("Drag to compare", 0, 100, 50, label_visibility="collapsed")
        compare_img = make_compare_image(original, passes["relit"], split)
        st.image(compare_img, channels="BGR", use_column_width=True)
        cl, cr = st.columns(2)
        cl.caption("◂ Original")
        cr.markdown("<div style='text-align:right'>Result ▸</div>", unsafe_allow_html=True)

    st.subheader("All passes")
    cols = st.columns(len(passes) or 1)
    for col, name in zip(cols, passes.keys()):
        with col:
            st.image(passes[name], channels="BGR", use_column_width=True)
            st.caption(PASS_LABELS.get(name, name))

    with st.expander("Stage log"):
        for line in st.session_state["log"]:
            if "unavailable" in line or "failed" in line:
                st.markdown(f":orange[{line}]")
            else:
                st.text(line)
