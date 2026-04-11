"""
RETINA-Q Streamlit Demo
Run with:
    streamlit run backend/streamlit_demo.py
"""
import base64
import io
import os
import sys

import streamlit as st
from PIL import Image

# Ensure app.* imports resolve when running from the project root
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

# ──────────────────────────────────────────────────────────────
# Page config — must be the FIRST streamlit call
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RETINA-Q Demo",
    page_icon="🔬",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# Lazy model loading (cached so models load only once)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models… this may take a moment.")
def get_inference_fns():
    from app.services.inference import run_oct_inference, run_fundus_inference
    return run_oct_inference, run_fundus_inference


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def pil_to_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG")
    return buf.getvalue()


def b64_to_pil(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


# ──────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────
st.title("🔬 RETINA-Q")
st.subheader("Hybrid Quantum-Classical Retinal Disease Diagnosis")
st.markdown(
    "Upload a retinal image below and run inference using our "
    "PennyLane quantum circuits."
)
st.divider()

tab_oct, tab_fundus = st.tabs(["🧬 OCT Classification", "👁️ Fundus Classification"])

# ── OCT Tab ───────────────────────────────────────────────────
with tab_oct:
    st.markdown("### OCT Quantum Model — Normal vs CSR")
    st.caption("8-qubit variational quantum circuit with 8 entangling layers.")

    col_left, col_right = st.columns(2)

    with col_left:
        uploaded_oct = st.file_uploader(
            "Upload an OCT scan (JPEG / PNG)",
            type=["jpg", "jpeg", "png"],
            key="oct_uploader",
        )
        if uploaded_oct:
            pil_oct = Image.open(uploaded_oct)
            st.image(pil_oct, caption="Uploaded OCT Scan", use_container_width=True)

    with col_right:
        if uploaded_oct and st.button("▶ Run OCT Inference", type="primary", key="oct_btn"):
            run_oct, _ = get_inference_fns()
            with st.spinner("Running quantum inference…"):
                try:
                    result = run_oct(pil_to_bytes(pil_oct))

                    pred = result.get("prediction", "Unknown")
                    conf = result.get("confidence", 0.0) * 100

                    if pred == "CSR":
                        st.error(f"🔴 **Prediction: {pred}** — Confidence: {conf:.2f}%")
                    else:
                        st.success(f"🟢 **Prediction: {pred}** — Confidence: {conf:.2f}%")

                    # Heatmap
                    if result.get("heatmap_base64"):
                        st.markdown("**Explainability Heatmap**")
                        st.image(
                            b64_to_pil(result["heatmap_base64"]),
                            use_container_width=True,
                        )

                    # Feature importance
                    if result.get("feature_importance"):
                        with st.expander("Feature Importance Details"):
                            st.json(result["feature_importance"])

                except Exception as e:
                    st.error(f"Inference failed: {e}")

# ── Fundus Tab ────────────────────────────────────────────────
with tab_fundus:
    st.markdown("### Fundus Hybrid Model — Healthy vs CSCR")
    st.caption("EfficientNet-B0 backbone + 4-qubit quantum layer (6 variational layers).")

    col_left, col_right = st.columns(2)

    with col_left:
        uploaded_fundus = st.file_uploader(
            "Upload a Fundus photograph (JPEG / PNG)",
            type=["jpg", "jpeg", "png"],
            key="fundus_uploader",
        )
        if uploaded_fundus:
            pil_fundus = Image.open(uploaded_fundus)
            st.image(pil_fundus, caption="Uploaded Fundus Image", use_container_width=True)

        run_seg = st.checkbox("Also run U-Net Macular Segmentation", value=False)

    with col_right:
        if uploaded_fundus and st.button("▶ Run Fundus Inference", type="primary", key="fundus_btn"):
            _, run_fundus = get_inference_fns()
            with st.spinner("Running hybrid quantum inference…"):
                try:
                    result = run_fundus(pil_to_bytes(pil_fundus), run_segmentation=run_seg)

                    pred = result.get("prediction", "Unknown")
                    conf = result.get("confidence", 0.0) * 100

                    if pred == "CSCR":
                        st.error(f"🔴 **Prediction: {pred}** — Confidence: {conf:.2f}%")
                    else:
                        st.success(f"🟢 **Prediction: {pred}** — Confidence: {conf:.2f}%")

                    # Grad-CAM
                    if result.get("gradcam_base64"):
                        st.markdown("**Grad-CAM Hotspots**")
                        st.image(
                            b64_to_pil(result["gradcam_base64"]),
                            use_container_width=True,
                        )

                    # Segmentation
                    if result.get("segmentation"):
                        seg = result["segmentation"]
                        st.markdown("**Macular Segmentation Overlay**")
                        if seg.get("overlay_base64"):
                            st.image(
                                b64_to_pil(seg["overlay_base64"]),
                                use_container_width=True,
                            )
                        area = seg.get("mask_area_ratio", 0.0) * 100
                        st.caption(f"Mask area: {area:.2f}% of image")

                except Exception as e:
                    st.error(f"Inference failed: {e}")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ This is an educational research demo. "
    "Not a medical device. Always consult a qualified ophthalmologist."
)
