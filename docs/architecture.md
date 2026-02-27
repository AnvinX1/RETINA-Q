# System Architecture
## RETINA-Q

---

# Pipeline Overview

User Upload
→ Image Type Detection
→ OCT Quantum Pipeline OR Fundus Quantum-EfficientNet
→ Conditional Segmentation
→ Explainability + Report

---

# OCT Pipeline

1. Feature Extraction (64 statistical features)
2. 8-Qubit Quantum Circuit
3. Classical Dense Layers
4. Binary Output

---

# Fundus Pipeline

1. EfficientNet-B0 Backbone
2. Adaptive Pooling
3. 4-Qubit Quantum Layer
4. Feature Concatenation
5. Binary Output

---

# Segmentation Pipeline

1. Green Channel Extraction
2. CLAHE Enhancement
3. U-Net Inference
4. Morphological Post-Processing
5. Mask Output

---

# Explainability

OCT:
- Feature importance mapping

Fundus:
- Grad-CAM on final convolution block

---