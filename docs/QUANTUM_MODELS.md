# RETINA-Q: Quantum Model Specifications

The RETINA-Q system utilizes a **dual-modality diagnostic pipeline**. By combining two distinct imaging techniques—OCT and Fundus photography—the system achieves a comprehensive clinical assessment of the retina, covering both surface pathology and deep structural changes.

---

## 1. OCT Model (Quantum-Classical Hybrid)
**Focus:** High-resolution cross-sectional depth analysis.

### Architecture
- **Inference Engine:** 8-Qubit PennyLane Variational Circuit.
- **Circuit Depth:** 8 Variational Layers (Angle Embedding + CNOT Entanglement).
- **Feature Extraction:** Advanced intensity profile analysis (64-dimensional feature vector).
- **Output Head:** Classical linear classifier (8 → 32 → 16 → 1).

### Purpose
Optical Coherence Tomography (OCT) is the "ultrasound of the eye." This model is specifically tuned to detect **Central Serous Retinopathy (CSR)** by analyzing sub-retinal fluid accumulation and retinal layer detachment that are often invisible on surface-level photography.

---

## 2. Fundus Model (Quantum-Classical Hybrid)
**Focus:** Wide-field retinal surface photography.

### Architecture
- **Backbone:** EfficientNet-B0 (Classical Convolutional Neural Network).
- **Quantum Layer:** 4-Qubit Variational Circuit (integrated into the feature bottleneck).
- **Circuit Depth:** 6 Variational Layers (RY/RZ Rotations + Ring Entanglement).
- **Input:** 224x224 RGB Fundus images.
- **Output Head:** Classical linear classifier (8 → 32 → 16 → 1).

### Purpose
Fundus photography provides a broad "top-down" view of the retina. This model identifies **Central Serous Chorioretinopathy (CSCR)** by detecting pigmentary changes, leakage patterns, and overall vascular health across a wide area of the posterior pole.

---

## 3. Comparison Matrix

| Feature | OCT Model | Fundus Model |
| :--- | :--- | :--- |
| **Input Type** | 64-dim Feature Vector (from depth scans) | 224x224 Color Image (Surface) |
| **Quantum Capacity**| 8 Qubits | 4 Qubits |
| **Circuit Layers**  | 8 Layers | 6 Layers |
| **Backbone** | Custom Intensity Feature Extractor | EfficientNet-B0 (Pretrained) |
| **Clinical Value**  | Sub-retinal fluid & depth metrics | Surface markers & vascular patterns |

---

## 4. Why Two Models? (The Clinical Synergy)

In modern ophthalmology, a single image type is rarely enough for a definitive diagnosis. RETINA-Q replicates the professional clinical workflow:

1.  **Macro vs. Micro**: Fundus images provide the **Macro** context (where is the lesion?), while OCT provides the **Micro** detail (how deep is the fluid?).
2.  **Redundancy & Confidence**: If both models detect pathology, the diagnostic confidence score is significantly higher. 
3.  **Complete Triage**: Some conditions (like early CSCR) may show pigment changes on Fundus before fluid is visible on OCT, and vice versa. Using both models ensures no pathology is missed during the initial triage.
4.  **Quantum Advantage**: By using different qubit counts (8 vs 4), we optimize the quantum representation for the specific complexity of the data—complex depth profiles vs. high-dimensional image features.
