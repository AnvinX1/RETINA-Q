# RETINA-Q Dataset Overview

## 1. Datasets

| Dataset | Source | Size | Period | Task |
|:--------|:-------|:-----|:-------|:-----|
| Kermany2018 OCT | `paultimothymooney/kermany2018` (Kaggle) | ~84 K images | 2018 | Normal vs. CSR binary classification |
| ODIR-5K Fundus | `andrewmvd/ocular-disease-recognition-odir5k` (Kaggle) | 5 000 patients | 2019 | Healthy vs. CSCR binary classification |

---

## 2. Dataset Attributes

### 2.1 Kermany2018 OCT Dataset

| Attribute | Value |
|:----------|:------|
| **Total images** | ~84 000 |
| **Image format** | JPEG |
| **Colour space** | Grayscale (converted from BGR at load time) |
| **Original resolution** | Variable (all resized to 224×224 during feature extraction) |
| **Classes (raw)** | NORMAL, CNV, DME, DRUSEN |
| **Binary label mapping** | `NORMAL → 0` (no disease), `CNV / DME / DRUSEN → 1` (CSR/Disease) |
| **Approximate class split** | ~26 K NORMAL · ~37 K CNV · ~11 K DME · ~8 K DRUSEN |
| **Provided splits** | `train/` and `test/` subdirectories per class |
| **Train / Val split used** | 80 % train · 20 % val (stratified, `random_state=42`) |
| **Training batch size** | 32 |
| **Supported file types** | `.jpeg`, `.jpg`, `.png`, `.tiff` |

**Directory structure expected:**
```
data/oct/
└── OCT2017/
    ├── train/
    │   ├── NORMAL/
    │   ├── CNV/
    │   ├── DME/
    │   └── DRUSEN/
    └── test/
        ├── NORMAL/
        ├── CNV/
        ├── DME/
        └── DRUSEN/
```

**Training-time data augmentation:**
| Transform | Parameters |
|:----------|:-----------|
| RandomHorizontalFlip | p = 0.5 |
| RandomVerticalFlip | p = 0.5 |
| RandomRotation | ±15° |
| ColorJitter | brightness ±0.2, contrast ±0.2 |

---

### 2.2 ODIR-5K Fundus Dataset

| Attribute | Value |
|:----------|:------|
| **Total patients** | 5 000 (left + right eye images → up to 10 000 images) |
| **Image format** | JPEG |
| **Colour space** | RGB (3-channel) |
| **Input resolution for model** | 224×224 px (resized during preprocessing) |
| **Metadata file** | `full_df.csv` (one row per image) |
| **Image directory** | `preprocessed_images/` |
| **Class columns in CSV** | `N` (Normal), `D` (Diabetes), `G` (Glaucoma), `C` (Cataract), `A` (AMD), `H` (Hypertension), `M` (Myopia), `O` (Other) |
| **Binary label mapping** | `N == 1 → 0` (Healthy), any other condition → `1` (Disease/CSR) |
| **Train / Val split used** | 80 % train · 20 % val (stratified, `random_state=42`) |
| **Training batch size** | 16 (dropped last incomplete batch) |
| **Optimiser** | AdamW (`lr=1e-4`, `weight_decay=1e-4`) |
| **LR schedule** | CosineAnnealingLR (`T_max = epochs`) |

**CSV schema (key columns):**
| Column | Type | Description |
|:-------|:-----|:------------|
| `filename` | string | Image filename relative to `preprocessed_images/` |
| `N` | int (0/1) | Normal eye flag |
| `D` | int (0/1) | Diabetic retinopathy flag |
| `G` | int (0/1) | Glaucoma flag |
| `C` | int (0/1) | Cataract flag |
| `A` | int (0/1) | Age-related macular degeneration flag |
| `H` | int (0/1) | Hypertensive retinopathy flag |
| `M` | int (0/1) | Pathologic myopia flag |
| `O` | int (0/1) | Other disease flag |
| `binary_label` | int (0/1) | Derived: `0` = Healthy, `1` = Disease |

**Training-time data augmentation:**
| Transform | Parameters |
|:----------|:-----------|
| Resize | 224×224 |
| RandomHorizontalFlip | p = 0.5 |
| RandomVerticalFlip | p = 0.5 |
| RandomRotation | ±15° |
| ColorJitter | brightness ±0.2, contrast ±0.2, saturation ±0.1 |
| Normalize | mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] (ImageNet) |

**Validation / inference preprocessing (no augmentation):**
| Transform | Parameters |
|:----------|:-----------|
| Resize | 224×224 |
| Normalize | mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225] (ImageNet) |

---

## 3. Features Extracted

### OCT (64-dimensional vector)
| # | Group | Count | Method / Formula | Indices |
|:-:|:------|:-----:|:-----------------|:--------|
| 1 | Gradient magnitude | 16 | Sobel operator `|G| = √(Gx² + Gy²)` applied to image; magnitude array split into 4 horizontal bands → per band: **mean, std, max, P75** | 0–15 |
| 2 | Histogram distribution | 16 | Normalised 256-bin pixel histogram split into 4 quartile ranges → per range: **sum, mean, std, argmax** | 16–31 |
| 3 | Local Binary Pattern (LBP) | 16 | Uniform LBP (radius=3, points=24) → 26-bin density histogram, **first 16 bins** retained | 32–47 |
| 4 | Texture variance | 8 | Image split into a **2×4 spatial grid** (8 patches) → **variance** per patch | 48–55 |
| 5 | Statistical moments | 8 | Global pixel distribution: **mean, std, skewness, kurtosis, median, P25, P75, IQR** (P75−P25) | 56–63 |

**OCT feature vector layout:**
```
[0..15]  gradient_mean_band0, gradient_std_band0, gradient_max_band0, gradient_p75_band0,
         gradient_mean_band1, ..., gradient_p75_band3
[16..31] hist_sum_q0, hist_mean_q0, hist_std_q0, hist_argmax_q0,
         hist_sum_q1, ..., hist_argmax_q3
[32..47] lbp_bin_0, lbp_bin_1, ..., lbp_bin_15
[48..55] texture_var_patch(0,0), texture_var_patch(0,1), ..., texture_var_patch(1,3)
[56..63] mean, std, skewness, kurtosis, median, p25, p75, iqr
```

### Fundus (learned features — no manual extraction)
| Stage | Representation |
|:------|:---------------|
| Input | RGB image 224×224 px, ImageNet-normalised |
| EfficientNet-B0 backbone | 1 280-dimensional feature vector (after AdaptiveAvgPool2d + flatten) |
| Dense reduction | 1 280 → 64 (ReLU, Dropout) → 4 (Tanh) — classical quantum input |
| Quantum circuit output | 4 PauliZ expectation values ⟨Z⟩ ∈ [−1, +1] |
| Fusion | Concatenate classical 4-dim ‖ quantum 4-dim = **8-dim** |

Features are learned end-to-end by the **EfficientNet-B0** backbone; no hand-crafted extraction is applied.

---

## 4. Input / Output

| Pipeline | Input | Output |
|:---------|:------|:-------|
| OCT classifier | Grayscale image (any resolution) | `Normal` or `CSR` + confidence score |
| Fundus classifier | RGB image 224×224 px | `Healthy` or `CSCR` + confidence score |
| Macular segmentation | RGB image 224×224 px | Binary mask of macular region |

---

## 5. Process Flow

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

## 6. Key Algorithms & Formulas

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
| CLAHE (Contrast Limited AHE) | Fundus segmentation preprocessing — green-channel contrast enhancement |
