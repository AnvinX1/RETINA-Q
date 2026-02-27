# Product Requirements Document (PRD)
## RETINA-Q

---

# 1. Product Summary

RETINA-Q is a hybrid quantum-classical medical AI system for retinal disease
diagnosis using OCT and Fundus images with automated segmentation
and explainability.

---

# 2. Problem Statement

Retinal diseases such as:
- Central Serous Retinopathy (CSR)
- Central Serous Chorioretinopathy (CSCR)

Require:
- High precision detection
- Automated segmentation
- Interpretable AI outputs

---

# 3. Objectives

Primary Objectives:

- Implement 8-qubit OCT classifier
- Embed 4-qubit layer in EfficientNet
- Conditional segmentation workflow
- Grad-CAM explainability
- REST-based deployment

Performance Targets:

| Metric | Target |
|--------|--------|
| OCT Accuracy | ≥ 92% |
| Fundus Accuracy | ≥ 93% |
| Dice Score | ≥ 0.90 |
| ROC-AUC | ≥ 0.95 |
| Inference | < 2 sec |

---

# 4. Functional Requirements

## 4.1 OCT Classification

Input:
- Grayscale OCT image

Features (64):
- Gradient magnitude
- Histogram distribution
- LBP
- Texture variance
- Statistical moments

Quantum Layer:
- 8 qubits
- Angle embedding
- Basic entangler
- Parameter-shift gradients

Output:
- Binary prediction
- Confidence
- Feature heatmap

---

## 4.2 Fundus Classification

Backbone:
- EfficientNet-B0

Quantum Layer:
- 4 qubits
- RY & RZ rotations
- CNOT entanglement
- Feature concatenation

Output:
- Healthy / CSCR
- Confidence
- Grad-CAM
- Segmentation trigger

---

## 4.3 Macular Segmentation

Preprocessing:
- Green channel extraction
- CLAHE enhancement

Model:
- U-Net

Loss:
- BCE
- Dice
- Tversky (α=0.7, β=0.3)

Post-processing:
- Connected components
- Largest circular region
- Morphological refinement

---

# 5. Non-Functional Requirements

- Secure APIs
- Scalable architecture
- GPU acceleration
- Model versioning
- Logging & monitoring
- Low latency

---

# 6. Risk Management

| Risk | Mitigation |
|------|------------|
| Quantum instability | Gradient clipping |
| Overfitting | Data augmentation |
| High latency | TorchScript |
| Limited dataset | Transfer learning |

---