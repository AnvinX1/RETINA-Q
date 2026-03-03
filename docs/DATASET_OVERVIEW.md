# RETINA-Q Dataset Overview

## 1. Datasets

| Dataset | Source | Size | Period | Task |
|:--------|:-------|:-----|:-------|:-----|
| Kermany2018 OCT | `paultimothymooney/kermany2018` (Kaggle) | ~84 K images | 2018 | Normal vs. CSR binary classification |
| ODIR-5K Fundus | `andrewmvd/ocular-disease-recognition-odir5k` (Kaggle) | 5 000 patients | 2019 | Healthy vs. CSCR binary classification |

---

## 2. Features Extracted

### OCT (64-dimensional vector)
| Group | Count | Method / Formula |
|:------|:-----:|:-----------------|
| Gradient magnitude | 16 | Sobel operator: `|G| = √(Gx² + Gy²)` — mean, std, max, P75 per 4 row-bands |
| Histogram distribution | 16 | Normalised pixel histogram split into 4 quartile bins — sum, mean, std, argmax |
| Local Binary Pattern (LBP) | 16 | Uniform LBP (radius=3, points=24) → 26-bin histogram, truncated to 16 |
| Texture variance | 8 | Per-patch variance across a 2×4 spatial grid |
| Statistical moments | 8 | mean, std, skew, kurtosis, median, P25, P75, IQR = P75 − P25 |

### Fundus (raw pixels)
Features are learned end-to-end by the **EfficientNet-B0** backbone (no manual extraction).

---

## 3. Input / Output

| Pipeline | Input | Output |
|:---------|:------|:-------|
| OCT classifier | Grayscale image (any resolution) | `Normal` or `CSR` + confidence score |
| Fundus classifier | RGB image 224×224 px | `Healthy` or `CSCR` + confidence score |
| Macular segmentation | RGB image 224×224 px | Binary mask of macular region |

---

## 4. Process Flow

### A. OCT Pipeline
```
Raw image
  → Grayscale conversion (OpenCV)
  → Resize to 224×224
  → 64-dim feature vector (Gradient + Histogram + LBP + Texture + Moments)
  → Dense reduction: 64 → 32 (ReLU, BN, Dropout) → 8 (Tanh)
  → 8-qubit VQC: AngleEmbedding + 8× [RY/RZ + CNOT ring]
  → PauliZ expectation ×8
  → Dense head: 8 → 32 → 16 → 1 (logit)
  → Sigmoid → prediction + confidence
```

### B. Fundus Pipeline
```
RGB image (224×224)
  → EfficientNet-B0 backbone (pretrained, head removed) → 1 280-dim features
  → AdaptiveAvgPool2d → flatten
  → Dense reduction: 1 280 → 64 (ReLU, Dropout) → 4 (Tanh)
  → 4-qubit VQC: RY embedding + 6× [RY/RZ + CNOT ring]
  → PauliZ expectation ×4
  → Concat [classical 4-dim ‖ quantum 4-dim] = 8-dim
  → Dense head: 8 → 32 → 16 → 1 (logit)
  → Sigmoid → prediction + confidence
```

### C. Macular Segmentation Pipeline
```
RGB image (224×224)
  → U-Net encoder-decoder
  → Loss: BCE + Dice + Tversky  (penalizes false negatives on small ROI)
  → Binary segmentation mask (macular region)
```

---

## 5. Key Algorithms & Formulas

| Name | Usage |
|:-----|:------|
| Sobel gradient | OCT edge-based feature extraction |
| Uniform LBP | OCT texture encoding |
| AngleEmbedding (`qml.AngleEmbedding`) | Encode classical values as qubit rotation angles |
| Variational Quantum Circuit (VQC) | Trainable RY / RZ gates + CNOT entanglement ring |
| PauliZ expectation: `⟨Z⟩ = P(0) − P(1)` | Quantum measurement → classical scalar per qubit |
| Binary Cross-Entropy (BCE) | Classification loss |
| Dice Loss: `1 − 2|A∩B| / (|A|+|B|)` | Segmentation overlap loss |
| Tversky Loss | Weighted overlap loss (penalizes false negatives) |
| Backpropagation (`diff_method="backprop"`) | Gradient computation through quantum circuit via PyTorch autograd |
