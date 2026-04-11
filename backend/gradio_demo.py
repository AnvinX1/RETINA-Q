import gradio as gr
import torch
import numpy as np
import io
import base64
from PIL import Image

# Import backend dependencies
# We ensure we are running from the backend directory to resolve app.*
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.inference import model_manager, run_oct_inference, run_fundus_inference
from app.utils.image_processing import load_image_from_bytes


def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode a base64-encoded PNG/JPEG string into a PIL Image."""
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))

# Custom wrapper functions for Gradio
def predict_oct(image: Image.Image):
    if image is None:
        return None, "Please upload an OCT image first.", None
    
    # Convert PIL Image to bytes as our inference service expects it
    img_byte_arr = io.BytesIO()
    # Save as JPEG as that's standard for the backend
    # Convert to RGB if it's not (e.g. RGBA)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    try:
        # Run standard inference pipeline
        result = run_oct_inference(img_bytes)
        
        # Decode the heatmap from base64 if it exists to show it in Gradio
        heatmap_img = None
        if "heatmap_base64" in result and result["heatmap_base64"]:
            heatmap_img = decode_base64_image(result["heatmap_base64"])
            
        prediction_text = f"**Prediction:** {result.get('prediction', 'Unknown')}\n\n**Confidence:** {result.get('confidence', 0.0)*100:.2f}%"
        
        return prediction_text, heatmap_img, result.get("feature_importance", {})
        
    except Exception as e:
        return f"Error during OCT Inference: {str(e)}", None, None


def predict_fundus(image: Image.Image, run_segmentation: bool):
    if image is None:
        return None, "Please upload a Fundus image first.", None, None
    
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    try:
        # Run standard inference pipeline (defaults to not running segmentation for speed unless checked)
        result = run_fundus_inference(img_bytes, run_segmentation=run_segmentation)
        
        # Decode Grad-CAM
        gradcam_img = None
        if "gradcam_base64" in result and result["gradcam_base64"]:
            gradcam_img = decode_base64_image(result["gradcam_base64"])
            
        # Decode Segmentation
        segmentation_img = None
        if result.get("segmentation") and "overlay_base64" in result["segmentation"]:
            segmentation_img = decode_base64_image(result["segmentation"]["overlay_base64"])
            
        prediction_text = f"**Prediction:** {result.get('prediction', 'Unknown')}\n\n**Confidence:** {result.get('confidence', 0.0)*100:.2f}%"
        
        return prediction_text, gradcam_img, segmentation_img
        
    except Exception as e:
        return f"Error during Fundus Inference: {str(e)}", None, None


# Build Gradio UI
with gr.Blocks(title="RETINA-Q Demo", theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# RETINA-Q: Hybrid Quantum-Classical Retinal Disease Diagnosis")
    gr.Markdown("Upload retinal images to test the model inference using our PennyLane quantum circuits.")
    
    with gr.Tabs():
        # OCT Tab (8-qubit quantum model)
        with gr.TabItem("OCT Image Classification"):
            gr.Markdown("### OCT Quantum Model (Normal vs CSR)")
            gr.Markdown("*Leverages an 8-qubit variational quantum circuit.*")
            
            with gr.Row():
                with gr.Column():
                    oct_input = gr.Image(type="pil", label="Upload OCT Scan")
                    oct_btn = gr.Button("Run OCT Quantum Inference", variant="primary")
                
                with gr.Column():
                    oct_result = gr.Markdown(label="Prediction Result")
                    oct_heatmap = gr.Image(type="pil", label="Explainability Heatmap")
                    oct_features = gr.JSON(label="Feature Importance")
                    
            oct_btn.click(
                fn=predict_oct,
                inputs=oct_input,
                outputs=[oct_result, oct_heatmap, oct_features]
            )

        # Fundus Tab (Hybrid model + U-Net Segmentation)
        with gr.TabItem("Fundus Image Classification"):
            gr.Markdown("### Fundus Hybrid Model (Healthy vs CSCR)")
            gr.Markdown("*Leverages an EfficientNet backbone + 4-qubit quantum layer.*")
            
            with gr.Row():
                with gr.Column():
                    fundus_input = gr.Image(type="pil", label="Upload Fundus Photograph")
                    fundus_seg_checkbox = gr.Checkbox(label="Run Macular Segmentation (U-Net)", value=False)
                    fundus_btn = gr.Button("Run Fundus Hybrid Inference", variant="primary")
                
                with gr.Column():
                    fundus_result = gr.Markdown(label="Prediction Result")
                    fundus_gradcam = gr.Image(type="pil", label="Grad-CAM Hotspots")
                    fundus_seg_out = gr.Image(type="pil", label="Segmentation Overlay")
                    
            fundus_btn.click(
                fn=predict_fundus,
                inputs=[fundus_input, fundus_seg_checkbox],
                outputs=[fundus_result, fundus_gradcam, fundus_seg_out]
            )

if __name__ == "__main__":
    # Launch on 0.0.0.0 to allow access from local machine or container
    print("Starting RETINA-Q Gradio Demo...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
