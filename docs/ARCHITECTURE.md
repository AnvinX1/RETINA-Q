# RETINA-Q System Architecture

RETINA-Q is a hybrid classical-quantum medical imaging system designed to evaluate the practical applications of Variational Quantum Circuits (VQCs) in high-stakes environments like retinal disease diagnosis.

This document serves as a technical deep-dive into the models and architecture of the RETINA-Q system.

---

## 1. System Topology

The system is broken down into three primary micro-services:

1.  **Backend (FastAPI & PyTorch/PennyLane)**: The core AI engine. It exposes RESTful endpoints for image ingestion, preprocessing, and model inference. It runs the heavy PyTorch tensors and simulates the PennyLane quantum node environments.
2.  **Frontend (Next.js & React)**: A modern, responsive dashboard designed with a minimalist black-and-white aesthetic (Shadcn/UI). It provides the clinician interface for uploading scans and viewing Grad-CAM explainability maps.
3.  **Database (PostgreSQL & Redis)**: Used for storing structured prediction logs and asynchronous job states (Celery).

---

## 2. Model Architectures

### A. Macular Segmentation (Classical)
-   **Architecture**: Modified U-Net
-   **Input**: `224x224x3` RGB images.
-   **Task**: Generating binary masks to highlight the macular region.
-   **Key Innovation**: Instead of standard BCE loss, the model is trained on a composite **BCE + Dice + Tversky** loss function. This heavily penalizes false negatives, which is crucial because the macula is a very small region of interest in a large fundus scan. 
-   **Performance**: `Dice Score: 0.9663`

### B. Fundus Classification (Hybrid Quantum-Classical)
-   **Architecture**: `EfficientNet-B0` + `4-Qubit VQC`
-   **Task**: Classify Fundus imagery (Healthy vs. Central Serous Chorioretinopathy).
-   **Pipeline**:
    1.  **Feature Extraction**: The input image passes through a pre-trained `EfficientNet-B0` backbone, bypassing the classical classifier head to yield a `1280`-dimensional feature vector.
    2.  **Dimensionality Reduction**: A dense NN sequence reduces these `1280` features down to `4` values to match the qubit count.
    3.  **Quantum Node**: The reduced features undergo Angle Embedding (RY rotations) across 4 qubits. They then pass through 6 variational layers of trainable RY/RZ rotations intertwined with CNOT entanglement rings. 
    4.  **Measurement**: The circuit measures the expectation value of `PauliZ` on each of the 4 qubits.
    5.  **Classification Head**: The `4` quantum outputs are concatenated with the `4` classical reduced inputs (`8`-dim total) and passed through a final dense layer for binary logits.
-   **Gradient Strategy**: `diff_method="backprop"`

### C. OCT Classification (Fully Quantum Domain)
-   **Architecture**: `8-Qubit VQC`
-   **Task**: Classify OCT cross-sectional scans (Normal vs. CSR).
-   **Pipeline**:
    1.  **Classical Pre-Processing**: Instead of a CNN, classical statistical features (e.g., intensity, variations) are extracted using OpenCV, resulting in a `64`-dim sparse vector.
    2.  **Reduction**: Reduced from `64` to `8` features, then bounded using `Tanh()` to ensure they map perfectly to rotation angles `[-1, 1]`.
    3.  **Quantum Node**: The 8 features undergo Angle Embedding across 8 qubits. They then pass through completely un-parameterized basic entangler layers, followed by 8 dense variational layers (RY/RZ + CNOT rings).
    4.  **Measurement**: `PauliZ` expectations on all 8 qubits.
    5.  **Classification**: The `8` measurements pass through a dense classical head to yield binary logits.
-   **Gradient Strategy**: `diff_method="backprop"`

---

## 3. The `diff_method` Quantum Barrier

Initially, the PennyLane `qnode` decorators were configured with `diff_method="parameter-shift"`. 
-   **The Problem**: Parameter-shift rules evaluate the circuit multiple times for *every single trainable parameter* to compute analytical gradients. For an 8-qubit, 8-layer circuit, this results in thousands of circuit evaluations *per batch*. This caused an extreme computational bottleneck.
-   **The Solution**: By switching to `diff_method="backprop"` and utilizing `default.qubit` (a state-vector simulator), PennyLane delegates the gradient tracking to PyTorch's native `autograd`. The simulation remains entirely within standard tensor memory on the GPU, exponentially speeding up step times to match classical DL speeds.
