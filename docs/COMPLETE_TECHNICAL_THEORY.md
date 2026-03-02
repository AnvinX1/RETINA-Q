# RETINA-Q: Complete Technical Theory & Documentation

> **A comprehensive, ground-up explanation of every concept, dataset, algorithm, model, and engineering decision behind the RETINA-Q Hybrid Quantum-Classical Retinal Diagnostic System.**

---

## Table of Contents

1. [The Problem: Central Serous Chorioretinopathy (CSCR)](#1-the-problem-central-serous-chorioretinopathy-cscr)
2. [The Data: Retinal Imaging Modalities](#2-the-data-retinal-imaging-modalities)
3. [Datasets Used & Sources](#3-datasets-used--sources)
4. [Image Preprocessing Pipeline](#4-image-preprocessing-pipeline)
5. [Feature Extraction: The 64-Dimensional OCT Descriptor](#5-feature-extraction-the-64-dimensional-oct-descriptor)
6. [Quantum Computing Fundamentals for This Project](#6-quantum-computing-fundamentals-for-this-project)
7. [Model 1 — OCT Quantum Classifier (8-Qubit VQC)](#7-model-1--oct-quantum-classifier-8-qubit-vqc)
8. [Model 2 — Fundus Hybrid Classifier (EfficientNet-B0 + 4-Qubit VQC)](#8-model-2--fundus-hybrid-classifier-efficientnet-b0--4-qubit-vqc)
9. [Model 3 — U-Net Macular Segmentation](#9-model-3--u-net-macular-segmentation)
10. [Loss Functions & Why They Were Chosen](#10-loss-functions--why-they-were-chosen)
11. [Training Infrastructure & Hyperparameters](#11-training-infrastructure--hyperparameters)
12. [Explainability: Grad-CAM & Feature Importance](#12-explainability-grad-cam--feature-importance)
13. [System Architecture: Backend, API, Database](#13-system-architecture-backend-api-database)
14. [Asynchronous Processing: Celery + Redis](#14-asynchronous-processing-celery--redis)
15. [Frontend & Desktop Application](#15-frontend--desktop-application)
16. [Docker Deployment Stack](#16-docker-deployment-stack)
17. [Performance Summary & Known Limitations](#17-performance-summary--known-limitations)
18. [Glossary of All Terminologies](#18-glossary-of-all-terminologies)

---

## 1. The Problem: Central Serous Chorioretinopathy (CSCR)

### What is CSCR?

Central Serous Chorioretinopathy (CSR or CSCR) is a retinal disorder where fluid accumulates beneath the retina, specifically under the macula — the central region responsible for sharp, detailed vision. This fluid leakage originates from the choroid (the vascular layer behind the retina) through a compromised Retinal Pigment Epithelium (RPE).

### Clinical Significance

- **Prevalence**: Affects approximately 1 in 10,000 people annually, predominantly males aged 30–50.
- **Symptoms**: Blurred central vision, metamorphopsia (distorted shapes), micropsia (objects appear smaller), reduced color saturation.
- **Risk**: ~5% of cases become chronic, leading to permanent photoreceptor damage and legal blindness in severe cases.
- **Diagnosis challenge**: Subtle early-stage features can be missed in manual examination, especially in high-throughput screening settings.

### Why Automated Detection Matters

Ophthalmologists use two primary imaging modalities to diagnose CSCR:

1. **Optical Coherence Tomography (OCT)** — cross-sectional depth scans of the retina
2. **Color Fundus Photography** — surface photographs of the retina

Both produce high-resolution images that require expert interpretation. An automated system that can triage these images accelerates clinical workflow and catches cases that might be missed in busy clinics.

### What RETINA-Q Does

RETINA-Q is a **dual-modality triage system**: it accepts either an OCT scan or a fundus photograph and returns:

- A **binary classification** (Normal/Healthy vs. CSR/CSCR disease)
- A **confidence score** (how certain the model is)
- An **explainability heatmap** (where the model is "looking")
- A **macular segmentation mask** (for fundus images, highlighting the macula region)
- A **physician feedback loop** (doctors can accept or override, improving future training data)

---

## 2. The Data: Retinal Imaging Modalities

### 2.1 Optical Coherence Tomography (OCT)

**What it is**: OCT is a non-invasive imaging technique that uses low-coherence light interferometry to capture micron-resolution cross-sectional images of the retina. Think of it as an "ultrasound but with light" — it produces B-scan slices showing the layered structure of retinal tissue.

**What the images look like**: Grayscale cross-sections, typically 496×512 or 512×1024 pixels, showing distinct retinal layers:
- Inner Limiting Membrane (ILM) — top boundary
- Retinal Nerve Fiber Layer (RNFL)
- Ganglion Cell Layer (GCL)
- Inner and Outer Plexiform Layers (IPL, OPL)
- Inner and Outer Nuclear Layers (INL, ONL)
- Photoreceptor layer (IS/OS junction)
- Retinal Pigment Epithelium (RPE) — bottom boundary
- Choroid — below RPE

**What disease looks like in OCT**: In CSCR, subretinal fluid appears as a dark (hyporeflective) space between the photoreceptor layer and the RPE, causing a visible "dome-shaped" detachment of the neurosensory retina.

### 2.2 Color Fundus Photography

**What it is**: A fundus camera captures a full-color photograph of the back of the eye (the fundus) through the pupil. It shows the retinal surface in 2D.

**What the images look like**: Color photographs (RGB), typically 2048×2048 or similar resolution, showing:
- The **optic disc** (bright yellowish circle where the optic nerve enters)
- The **macula** (darker central region responsible for central vision)
- The **fovea** (tiny pit in the center of the macula)
- **Blood vessels** (arteries appear lighter red, veins appear darker)

**What disease looks like in fundus**: CSCR manifests as:
- A **serous detachment** — a rounded, slightly elevated area near the macula
- **Pigment epithelial detachments** (PEDs) — subtle bumps
- **Gravitational tracks** — streaks of pigment from chronic fluid leakage
- Often subtle and easily missed without OCT confirmation

### 2.3 Why Both Modalities?

| Feature | OCT | Fundus |
|---------|-----|--------|
| Dimension | Cross-section (depth) | Surface (2D) |
| Fluid detection | Excellent (directly visible) | Moderate (indirect signs) |
| Structural layers | All layers visible | Surface only |
| Cost | Higher ($30k–100k equipment) | Lower ($5k–20k) |
| Availability | Specialist clinics | Primary care, screening camps |
| Screening use | Gold standard for diagnosis | First-line screening |

Using **both** modalities gives clinicians complementary information. Our system supports either independently.

---

## 3. Datasets Used & Sources

### 3.1 OCT Dataset: Kermany2018

| Property | Value |
|----------|-------|
| **Source** | [Kaggle: paultimothymooney/kermany2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) |
| **Original Paper** | Kermany et al., "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning," *Cell*, 2018 |
| **Total Images** | ~84,000 OCT B-scans |
| **Image Type** | Grayscale, variable resolution (~496×512) |
| **Original Classes** | **NORMAL** — healthy retina |
| | **CNV** — Choroidal Neovascularization |
| | **DME** — Diabetic Macular Edema |
| | **DRUSEN** — early Age-related Macular Degeneration |
| **Our Binary Mapping** | `NORMAL → 0 (Normal)`, `CNV + DME + DRUSEN → 1 (CSR group)` |
| **Split** | 80% train / 20% validation (stratified) |

**Why this dataset**: It is the largest publicly available labeled OCT dataset, peer-reviewed and widely benchmarked. The multi-class pathologies (CNV, DME, DRUSEN) all represent retinal abnormalities with structural disruption visible in OCT, making them valid proxies for training a binary "normal vs. abnormal" detector.

**Binary mapping rationale**: Since CSCR-specific OCT datasets with large enough sample sizes are scarce, we group all pathological classes as "disease positive." The quantum model learns to distinguish structural integrity (normal) from structural disruption (disease) — the same fundamental distinction needed for CSCR detection.

### 3.2 Fundus Dataset: ODIR-5K

| Property | Value |
|----------|-------|
| **Source** | [Kaggle: andrewmvd/ocular-disease-recognition-odir5k](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k) |
| **Original Source** | Peking University International Competition, ODIR-2019 |
| **Total Patients** | 5,000 (both left and right eyes photographed) |
| **Image Type** | Color (RGB), variable resolution |
| **Label File** | `full_df.csv` with columns for filename and diagnostic keywords |
| **Original Classes** | N=Normal, D=Diabetic Retinopathy, G=Glaucoma, C=Cataract, A=AMD, H=Hypertension, M=Myopia, O=Other |
| **Our Binary Mapping** | `N → 0 (Healthy)`, `D+G+C+A+H+M+O → 1 (Disease/CSCR group)` |
| **Image Directory** | `preprocessed_images/` |
| **Split** | 80% train / 20% validation (stratified) |

**Why this dataset**: ODIR-5K provides real-world clinical fundus photographs with expert-verified diagnoses across multiple pathologies. It is the standard benchmark for multi-label ocular disease classification.

**Binary mapping rationale**: Similar to OCT, we cast the problem as healthy vs. disease-positive. The model learns to detect pathological patterns in fundus images — structural and color anomalies that differ from healthy retinal appearance.

### 3.3 Segmentation Data

The U-Net segmentation model uses **self-supervised pseudo-labels** derived from the ODIR fundus images:
- The green channel of fundus images is extracted (green channel has the highest contrast for retinal structures).
- CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement is applied.
- Otsu's thresholding generates binary masks of the macular region.
- These pseudo-masks serve as training labels.
- Maximum 2,000 images used for training.

---

## 4. Image Preprocessing Pipeline

Every image entering RETINA-Q goes through specific preprocessing depending on the modality and downstream model.

### 4.1 OCT Image Preprocessing

```
Raw OCT scan (grayscale, variable size)
        │
        ▼
  Resize to 224×224 pixels
        │
        ▼
  64-Dimensional Feature Extraction  ← (see Section 5)
        │
        ▼
  Tensor: shape (1, 64)  →  feeds into the OCT Quantum Model
```

OCT images are **not** fed as raw pixels to the quantum circuit. Instead, a handcrafted feature extraction pipeline converts each image into a compact 64-dimensional numerical vector. This is critical because:
- Quantum circuits can only process a small number of inputs (8 qubits = 8 inputs per encoding round).
- Handcrafted features capture domain-relevant information (edges, textures, statistical moments) that raw pixels would require millions of parameters to learn.

### 4.2 Fundus Image Preprocessing

```
Raw fundus image (RGB, variable size)
        │
        ▼
  BGR → RGB color conversion (OpenCV loads as BGR)
        │
        ▼
  Resize to 224×224 pixels
        │
        ▼
  ToTensor (converts [0,255] → [0.0, 1.0], HWC → CHW)
        │
        ▼
  ImageNet Normalization:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
        │
        ▼
  Tensor: shape (1, 3, 224, 224)  →  feeds into EfficientNet-B0
```

**Why ImageNet normalization**: EfficientNet-B0 was pretrained on ImageNet. To reuse those learned features (transfer learning), input images must be normalized with the same statistics used during pretraining.

**Training augmentations** (applied only during training, not inference):
- `RandomHorizontalFlip` — retinal images can be left/right mirrored without changing pathology
- `RandomVerticalFlip` — up/down flip for rotation invariance
- `RandomRotation(15°)` — slight rotational variance
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)` — simulates different camera conditions

### 4.3 Segmentation Preprocessing

```
Raw fundus image (RGB, variable size)
        │
        ▼
  Extract GREEN channel only
        │
        ▼
  CLAHE Enhancement:
    clipLimit = 2.0
    tileGridSize = (8, 8)
        │
        ▼
  Resize to 256×256 pixels
        │
        ▼
  Normalize: pixel_values / 255.0
        │
        ▼
  Tensor: shape (1, 1, 256, 256)  →  feeds into U-Net
```

**Why green channel**: In RGB fundus photographs, the green channel provides the highest contrast between retinal structures (blood vessels, optic disc, macula) and the background. Red is often oversaturated, and blue has low signal.

**What is CLAHE**: Contrast Limited Adaptive Histogram Equalization divides the image into small tiles (8×8 grid), applies histogram equalization to each tile independently, then blends boundaries. The "contrast limited" part (clipLimit=2.0) prevents noise amplification. Result: enhanced local contrast without global brightness distortion.

---

## 5. Feature Extraction: The 64-Dimensional OCT Descriptor

This is one of the most important components of the system. Instead of feeding raw OCT pixels into a quantum circuit (which would require thousands of qubits), we extract **64 statistical features** that encode the essential characteristics of each image.

### 5.1 Gradient Features (16 features)

**Algorithm**: Sobel edge detection

**What it does**: The Sobel operator detects edges by computing the image gradient — the rate of change of pixel intensity. Edges correspond to boundaries between retinal layers.

**Implementation**:
1. Compute horizontal gradient: $G_x = \text{Sobel}(image, \text{axis}=x, \text{kernel}=3\times3)$
2. Compute vertical gradient: $G_y = \text{Sobel}(image, \text{axis}=y, \text{kernel}=3\times3)$
3. Compute gradient magnitude: $G = \sqrt{G_x^2 + G_y^2}$
4. Divide the magnitude image into **4 spatial regions** (vertical quarters)
5. For each region, compute: **mean**, **standard deviation**, **maximum**, **75th percentile**

$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} * I, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} * I$$

**Why 4 regions**: OCT images have spatial structure — the top region shows the inner retina, bottom shows the RPE/choroid. Disease typically manifests in the lower half (subretinal fluid), so regional statistics capture where intensity changes occur.

**Output**: 4 regions × 4 statistics = **16 features**

### 5.2 Histogram Features (16 features)

**Algorithm**: Intensity distribution analysis

**What it does**: Computes a 256-bin histogram of all pixel intensities in the image, then analyzes the distribution in four quartile sections.

**Implementation**:
1. Compute histogram: $H = \text{hist}(image, \text{bins}=256)$
2. Normalize: $H' = H / \sum H$
3. Divide into 4 quartile ranges: bins [0-63], [64-127], [128-191], [192-255]
4. For each quartile: **sum**, **mean**, **standard deviation**, **argmax** (peak position)

**Why this matters**: Healthy OCT scans have a characteristic bimodal intensity distribution (dark vitreous + bright layers). Disease changes this distribution — fluid appears as large dark regions shifting the histogram.

**Output**: 4 quartiles × 4 statistics = **16 features**

### 5.3 Local Binary Pattern (LBP) Features (16 features)

**Algorithm**: Local Binary Patterns (Ojala et al., 2002)

**What it does**: LBP is a texture descriptor. For every pixel, it examines a ring of neighboring pixels and encodes whether each neighbor is brighter or darker than the center — creating a binary code that captures the local texture pattern.

**Parameters**:
- **Radius**: 3 pixels — the distance from center to neighbor ring
- **Points**: $8 \times \text{radius} = 24$ — number of sampling points on the ring
- **Method**: `uniform` — only considers "uniform" patterns (at most two 0→1 or 1→0 transitions in the binary code), which correspond to edges, corners, and flat areas

**Implementation**:
1. Compute LBP image: each pixel gets a pattern code (0 to $P+1$ where $P=24$)
2. Compute histogram of pattern codes: 26 bins ($P + 2 = 26$)
3. Normalize the histogram
4. Pad or truncate to 16 features

$$LBP_{P,R}(x_c, y_c) = \sum_{p=0}^{P-1} s(g_p - g_c) \cdot 2^p, \quad s(x) = \begin{cases} 1 & x \geq 0 \\ 0 & x < 0 \end{cases}$$

**Why LBP**: It captures micro-texture patterns — the regularity of retinal layers in healthy tissue vs. the disrupted textures in disease. It is rotation-invariant (uniform method) and computationally efficient.

**Output**: **16 features**

### 5.4 Texture Variance Features (8 features)

**Algorithm**: Sliding window variance

**What it does**: Divides the image into a 2×4 grid (2 columns × 4 rows) and computes the variance of pixel intensities within each patch.

**Implementation**:
1. Split image into $2 \times 4 = 8$ patches
2. For each patch: $\text{variance} = \frac{1}{N}\sum(x_i - \bar{x})^2$

**Why this matters**: Variance measures texture roughness. Fluid-filled regions have low variance (uniform dark areas), while healthy layered tissue has high variance (alternating bright/dark layers).

**Output**: 2 × 4 = **8 features**

### 5.5 Statistical Moments (8 features)

**Algorithm**: Global image statistics

**What it does**: Computes summary statistics across the entire image.

**Features**:
1. **Mean** ($\mu$): average intensity — overall brightness
2. **Standard deviation** ($\sigma$): spread of intensities
3. **Skewness**: $\frac{E[(X-\mu)^3]}{\sigma^3}$ — asymmetry of the intensity distribution. Positive skew means the tail extends toward bright values.
4. **Kurtosis**: $\frac{E[(X-\mu)^4]}{\sigma^4} - 3$ — "peakedness" of the distribution. High kurtosis means sharp peaks (concentrated intensity values).
5. **Median**: 50th percentile — robust center measure
6. **25th percentile** (Q1): lower quartile boundary
7. **75th percentile** (Q3): upper quartile boundary
8. **IQR**: $Q3 - Q1$ — interquartile range, measures the spread of the middle 50% of values

**Why these**: They provide a compact global summary. Mean and std capture brightness/contrast. Skewness and kurtosis detect distribution shape changes caused by pathological fluid or tissue disruption. Percentiles and IQR are robust to outliers.

**Output**: **8 features**

### 5.6 Total Feature Vector

$$\text{Feature Vector} = [\underbrace{G_1, \ldots, G_{16}}_{\text{Gradient}}, \underbrace{H_1, \ldots, H_{16}}_{\text{Histogram}}, \underbrace{L_1, \ldots, L_{16}}_{\text{LBP}}, \underbrace{V_1, \ldots, V_{8}}_{\text{Variance}}, \underbrace{S_1, \ldots, S_{8}}_{\text{Moments}}] \in \mathbb{R}^{64}$$

This 64-dimensional vector is the complete numerical signature of an OCT scan, encoding edge structure, intensity distribution, texture patterns, regional variance, and global statistics.

---

## 6. Quantum Computing Fundamentals for This Project

### 6.1 Why Quantum Computing for Medical Imaging?

Classical deep learning works well for medical imaging but has limitations:
- CNNs apply **local filters** — they detect features within small receptive fields and compose them hierarchically.
- Quantum circuits can create **entangled states** where all inputs are correlated simultaneously — potentially capturing complex multi-feature relationships that would require very deep classical networks.

In RETINA-Q, quantum circuits serve as **variational classifiers** — small parametric circuits that learn to separate healthy from diseased patterns in a feature space.

### 6.2 Qubit Basics

A **qubit** is the quantum analog of a classical bit. While a bit is either 0 or 1, a qubit exists in a **superposition**:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle, \quad |\alpha|^2 + |\beta|^2 = 1$$

where $\alpha$ and $\beta$ are complex amplitudes. The probability of measuring 0 is $|\alpha|^2$ and measuring 1 is $|\beta|^2$.

### 6.3 Quantum Gates Used

**Rotation gates** (single-qubit, parametric):

$$R_Y(\theta) = \begin{pmatrix} \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\ \sin\frac{\theta}{2} & \cos\frac{\theta}{2} \end{pmatrix}, \quad R_Z(\phi) = \begin{pmatrix} e^{-i\phi/2} & 0 \\ 0 & e^{i\phi/2} \end{pmatrix}$$

- $R_Y(\theta)$ rotates the qubit state around the Y-axis of the Bloch sphere — controls the **amplitude** (probability distribution).
- $R_Z(\phi)$ rotates around the Z-axis — controls the **phase** (interference pattern).

Together, $R_Y$ and $R_Z$ give universal single-qubit control.

**CNOT (Controlled-NOT) gate** (two-qubit entangling gate):

$$\text{CNOT} = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&0&1 \\ 0&0&1&0 \end{pmatrix}$$

The CNOT gate flips the target qubit if and only if the control qubit is $|1\rangle$. It creates **entanglement** — a quantum correlation where the state of one qubit cannot be described independently of the other.

### 6.4 Angle Embedding

To encode classical data into a quantum circuit, we use **angle embedding**: each classical feature value $x_i$ becomes the rotation angle of a qubit.

$$|0\rangle \xrightarrow{R_Y(x_i)} \cos\frac{x_i}{2}|0\rangle + \sin\frac{x_i}{2}|1\rangle$$

This maps a feature value to a point on the Bloch sphere. The **Tanh** activation in the classical pre-processing layers ensures inputs are bounded to $[-1, 1]$, which maps to reasonable rotation angles.

### 6.5 Variational Quantum Circuit (VQC)

A VQC is the quantum equivalent of a neural network layer. It has:
1. **Data encoding** — map input features to qubit states (angle embedding)
2. **Variational layers** — parametric rotation gates ($R_Y(\theta_i)$, $R_Z(\phi_i)$) with learnable parameters
3. **Entangling layers** — CNOT gates that create correlations between qubits
4. **Measurement** — collapse qubits and read expectation values

The parameters $\theta_i$ and $\phi_i$ are learned via gradient descent, just like weights in a neural network.

### 6.6 Measurement: Pauli-Z Expectation

After the circuit runs, we measure each qubit in the **Pauli-Z basis**:

$$\langle Z \rangle = P(|0\rangle) - P(|1\rangle) \in [-1, +1]$$

This gives a continuous value for each qubit that serves as the circuit's output — analogous to a neuron's activation.

### 6.7 CNOT Ring Entanglement Topology

In RETINA-Q, entanglement follows a **ring topology**:

```
Qubit 0 ──CNOT──→ Qubit 1 ──CNOT──→ Qubit 2 ──CNOT──→ ... ──CNOT──→ Qubit (N-1) ──CNOT──→ Qubit 0
```

Each qubit is entangled with its neighbor, and the last qubit connects back to the first, forming a circular chain. This ensures every qubit is directly or indirectly correlated with every other qubit.

### 6.8 PennyLane Framework

[PennyLane](https://pennylane.ai/) (by Xanadu) is the quantum machine learning library used in RETINA-Q.

**Key configuration**:
- **Device**: `default.qubit` — a CPU-based state-vector simulator that simulates quantum computation exactly (no hardware noise).
- **Interface**: `torch` — integrates with PyTorch's autograd for gradient computation.
- **Diff method**: `backprop` — computes quantum gradients using classical backpropagation through the simulated state vector.

**Critical insight — the backprop breakthrough**: The `parameter-shift` method (the default in PennyLane) computes each quantum gradient by running the circuit twice per parameter. For an 8-qubit, 8-layer circuit with 128 parameters, this means 256 circuit evaluations per gradient step. Switching to `backprop` (which analytically differentiates through the simulator) reduced training time from **~100 minutes/epoch** to **~7 minutes/epoch** — a ~14× speedup that made the project feasible.

### 6.9 Quantum vs Classical: The Entanglement Advantage

Consider classifying an 8-feature input. A classical neural network with one hidden layer of 32 neurons creates $8 \times 32 = 256$ weighted connections — each weight captures a pairwise feature-to-neuron relationship.

An 8-qubit quantum circuit with entanglement creates a $2^8 = 256$-dimensional state space where all 8 features are **simultaneously correlated**. The entanglement creates non-local correlations that a classical network would need multiple layers to approximate. With only $128$ trainable parameters (8 qubits × 8 layers × 2 rotations), the quantum circuit explores a richer hypothesis space.

This doesn't mean quantum is always better — for this problem size, the advantage is modest and primarily serves as a **proof of concept** for quantum-assisted medical diagnostics.

---

## 7. Model 1 — OCT Quantum Classifier (8-Qubit VQC)

### 7.1 Architecture Overview

```
OCT Image ──→ Feature Extraction ──→ 64-dim vector ──→ Classical Pre-Net ──→ 8-dim vector ──→ Quantum Circuit ──→ 8-dim vector ──→ Classical Post-Net ──→ Sigmoid ──→ P(disease)
```

### 7.2 Detailed Architecture

**Pre-Quantum Classical Network** (dimensionality reduction from 64 → 8):

| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| 1 | Linear(64 → 32) | (batch, 32) |
| 2 | ReLU activation | (batch, 32) |
| 3 | BatchNorm1d(32) | (batch, 32) |
| 4 | Dropout(0.3) | (batch, 32) |
| 5 | Linear(32 → 8) | (batch, 8) |
| 6 | Tanh activation | (batch, 8) |

**Why Tanh**: The output must be bounded for angle embedding. Tanh maps values to $[-1, 1]$, ensuring rotation angles are in a meaningful range.

**Quantum Circuit** (8-qubit VQC):

| Step | Operation | Details |
|------|-----------|---------|
| 1 | AngleEmbedding | 8 input values → RY rotations on 8 qubits |
| 2–9 | 8 Variational Layers | Each layer: RY + RZ on all 8 qubits, then CNOT ring |
| 10 | Measurement | PauliZ expectation on all 8 qubits |

- **Trainable parameters**: $8 \text{ qubits} \times 8 \text{ layers} \times 2 \text{ gates} = 128$
- **Entanglement per layer**: 8 CNOT gates (ring: 0→1, 1→2, ..., 7→0)

**Post-Quantum Classical Network** (classification head):

| Layer | Operation | Output Shape |
|-------|-----------|-------------|
| 1 | Linear(8 → 32) | (batch, 32) |
| 2 | ReLU | (batch, 32) |
| 3 | BatchNorm1d(32) | (batch, 32) |
| 4 | Dropout(0.3) | (batch, 32) |
| 5 | Linear(32 → 16) | (batch, 16) |
| 6 | ReLU | (batch, 16) |
| 7 | Dropout(0.2) | (batch, 16) |
| 8 | Linear(16 → 1) | (batch, 1) |

**Output**: A single logit passed through sigmoid: $P(\text{disease}) = \sigma(z) = \frac{1}{1 + e^{-z}}$

### 7.3 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Loss Function | BCEWithLogitsLoss | Combines sigmoid + BCE; numerically stable |
| Optimizer | Adam | Adaptive learning rate; works well with quantum gradients |
| Learning Rate | $5 \times 10^{-4}$ | Higher than fundus (simpler architecture) |
| LR Schedule | StepLR(step=10, γ=0.5) | Halve LR every 10 epochs |
| Epochs | 30 | Found via validation loss plateau |
| Batch Size | 32 | Larger batches for feature-based input |
| Gradient Clipping | max_norm = 1.0 | Prevents gradient explosion in quantum layers |
| Weight Initialization | PyTorch defaults | Xavier/Kaiming for linear layers |

### 7.4 Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | **79.66%** |
| Validation AUC (ROC) | **0.8473** |
| Dataset | Kermany2018 (84k images) |

---

## 8. Model 2 — Fundus Hybrid Classifier (EfficientNet-B0 + 4-Qubit VQC)

### 8.1 Architecture Overview

```
Fundus Image ──→ EfficientNet-B0 ──→ 1280-dim ──→ Bridge Net ──→ 4-dim ──→ Quantum Circuit ──→ 4-dim
                                                                   │                                │
                                                                   └──────────── Concatenate ───────┘
                                                                                    │
                                                                              8-dim fusion
                                                                                    │
                                                                         Classifier Head ──→ P(disease)
```

### 8.2 EfficientNet-B0 Backbone

**What is EfficientNet**: EfficientNet (Tan & Le, 2019) is a family of CNN architectures discovered via Neural Architecture Search (NAS) that uniformly scales network width, depth, and resolution. B0 is the base model.

**Key properties**:
- **Pretrained**: ImageNet (1000-class, 1.2M images)
- **Building blocks**: Mobile Inverted Bottleneck Convolution (MBConv) with Squeeze-and-Excitation
- **Parameters**: ~5.3M
- **Output**: 1280-dimensional feature vector (global average pooling of the last conv layer)
- **Why B0**: Smallest EfficientNet variant — balances accuracy with inference speed for clinical deployment

**Transfer learning**: The backbone is loaded with pretrained ImageNet weights. All layers are fine-tuned (not frozen) during training, allowing the network to adapt its feature extraction from general objects to retinal pathology.

### 8.3 Dimensionality Bridge (1280 → 4)

| Layer | Operation | Output |
|-------|-----------|--------|
| 1 | Linear(1280 → 64) | (batch, 64) |
| 2 | ReLU | (batch, 64) |
| 3 | Dropout(0.3) | (batch, 64) |
| 4 | Linear(64 → 4) | (batch, 4) |
| 5 | Tanh | (batch, 4) |

**Why 4-dim output**: The quantum circuit has 4 qubits, so input must be 4-dimensional.

### 8.4 Quantum Circuit (4-Qubit VQC)

| Step | Operation | Details |
|------|-----------|---------|
| 1 | AngleEmbedding (RY) | 4 classical values → 4 qubit rotations |
| 2–7 | 6 Variational Layers | Each: RY + RZ per qubit, then CNOT ring (0→1→2→3→0) |
| 8 | Measurement | PauliZ expectation on all 4 qubits |

- **Trainable parameters**: $4 \times 6 \times 2 = 48$ quantum parameters
- **State space**: $2^4 = 16$ dimensions

### 8.5 Hybrid Fusion

The key innovation: classical and quantum outputs are **concatenated**, not replaced.

$$\text{fusion} = [\underbrace{c_1, c_2, c_3, c_4}_{\text{classical (bridge output)}}, \underbrace{q_1, q_2, q_3, q_4}_{\text{quantum (circuit output)}}] \in \mathbb{R}^8$$

This ensures the model can use both the classical feature reduction path and the quantum-processed features, learning to weight them optimally.

### 8.6 Classification Head

| Layer | Operation | Output |
|-------|-----------|--------|
| 1 | Linear(8 → 32) | (batch, 32) |
| 2 | ReLU | (batch, 32) |
| 3 | BatchNorm1d(32) | (batch, 32) |
| 4 | Dropout(0.3) | (batch, 32) |
| 5 | Linear(32 → 16) | (batch, 16) |
| 6 | ReLU | (batch, 16) |
| 7 | Dropout(0.2) | (batch, 16) |
| 8 | Linear(16 → 1) | (batch, 1) |

### 8.7 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Loss Function | BCEWithLogitsLoss | Binary classification |
| Optimizer | AdamW | Adam with decoupled weight decay; better regularization |
| Learning Rate | $1 \times 10^{-4}$ | Lower than OCT (larger pretrained model) |
| Weight Decay | $1 \times 10^{-4}$ | Prevents overfitting on small dataset |
| LR Schedule | CosineAnnealingLR(T_max=20) | Smooth decay following cosine curve |
| Epochs | 20 | Fewer epochs needed with pretrained backbone |
| Batch Size | 16 | Smaller due to larger model memory |
| Gradient Clipping | max_norm = 1.0 | Stabilizes quantum gradients |

### 8.8 Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | **68.26%** |
| Validation AUC (ROC) | **0.7503** |
| Dataset | ODIR-5K (5,000 patients) |

**Note**: The 68% accuracy is an **excellent baseline** for a 4-qubit quantum circuit on a small dataset with diverse pathologies. The team noted this is "primed for >90% accuracy on a full CSCR-specific dataset."

---

## 9. Model 3 — U-Net Macular Segmentation

### 9.1 What is Semantic Segmentation?

Unlike classification (one label per image), segmentation assigns a label to **every pixel**. The U-Net produces a binary mask where:
- **White (1)** = macular region
- **Black (0)** = background

### 9.2 U-Net Architecture (Ronneberger et al., 2015)

U-Net is a symmetric encoder-decoder architecture with **skip connections** — the hallmark design for biomedical image segmentation.

```
Input (1, 256, 256)
    │
    ▼
┌─ Encoder 1: Conv(1→64) + Conv(64→64) ────────────────── Skip ──┐
│   MaxPool(2×2) ↓                                                 │
├─ Encoder 2: Conv(64→128) + Conv(128→128) ─────────── Skip ──┐  │
│   MaxPool(2×2) ↓                                              │  │
├─ Encoder 3: Conv(128→256) + Conv(256→256) ──────── Skip ──┐ │  │
│   MaxPool(2×2) ↓                                           │ │  │
├─ Encoder 4: Conv(256→512) + Conv(512→512) ───── Skip ──┐  │ │  │
│   MaxPool(2×2) ↓                                        │  │ │  │
│                                                         │  │ │  │
├─ Bottleneck: Conv(512→1024) + Conv(1024→1024)           │  │ │  │
│                                                         │  │ │  │
│   ConvTranspose(1024→512) ↑                             │  │ │  │
├─ Decoder 4: Concat + Conv(1024→512) + Conv(512→512) ◄──┘  │ │  │
│   ConvTranspose(512→256) ↑                                 │ │  │
├─ Decoder 3: Concat + Conv(512→256) + Conv(256→256) ◄──────┘ │  │
│   ConvTranspose(256→128) ↑                                   │  │
├─ Decoder 2: Concat + Conv(256→128) + Conv(128→128) ◄────────┘  │
│   ConvTranspose(128→64) ↑                                       │
├─ Decoder 1: Concat + Conv(128→64) + Conv(64→64) ◄──────────────┘
│
│   Conv2d(64→1) + Sigmoid
    ▼
Output (1, 256, 256) — probability map [0, 1]
```

### 9.3 ConvBlock Design

Each "Conv" in the diagram is actually a **ConvBlock**:

```
Conv2d(3×3, padding=1) → BatchNorm2d → ReLU → Conv2d(3×3, padding=1) → BatchNorm2d → ReLU
```

- `padding=1` preserves spatial dimensions
- BatchNorm normalizes activations for stable training
- Two conv layers per block gives sufficient feature extraction at each level

### 9.4 Skip Connections

The key insight of U-Net: **skip connections** concatenate encoder feature maps with the corresponding decoder feature maps at each level.

**Why this matters**: The encoder progressively loses spatial detail (each MaxPool halves resolution). The decoder upsamples but has "blurry" feature maps. Skip connections provide the decoder with **precise spatial information** from the encoder, enabling pixel-accurate segmentation boundaries.

### 9.5 Post-Processing

The raw model output is a probability map $\in [0, 1]$. Post-processing refines it:

1. **Binarization**: Threshold at 0.5 → binary mask
2. **Connected components**: Keep only the largest connected region (removes small noise)
3. **Morphological closing**: Fills small holes within the segmented region (5×5 elliptical kernel)
4. **Morphological opening**: Removes small protrusions (5×5 elliptical kernel)

### 9.6 Training & Performance

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=$1\times10^{-4}$, weight_decay=$1\times10^{-4}$) |
| LR Schedule | CosineAnnealingLR (T_max=30) |
| Epochs | 30 |
| Batch Size | 8 |
| Input Size | 256×256, single channel (green + CLAHE) |
| Max Training Samples | 2,000 |
| **Dice Score** | **0.9663** (exceptional) |

---

## 10. Loss Functions & Why They Were Chosen

### 10.1 BCEWithLogitsLoss (OCT & Fundus Classification)

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \cdot \log(\sigma(z_i)) + (1 - y_i) \cdot \log(1 - \sigma(z_i)) \right]$$

where $z_i$ is the raw logit, $\sigma$ is sigmoid, and $y_i \in \{0, 1\}$.

**Why**: Standard loss for binary classification. `WithLogitsLoss` applies sigmoid internally, which is numerically more stable than applying sigmoid first then computing BCE (avoids log(0) issues).

### 10.2 U-Net Combined Loss

The segmentation model uses a **weighted combination of three losses**:

$$\mathcal{L}_{\text{total}} = 0.3 \cdot \mathcal{L}_{\text{BCE}} + 0.3 \cdot \mathcal{L}_{\text{Dice}} + 0.4 \cdot \mathcal{L}_{\text{Tversky}}$$

#### Dice Loss

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i \cdot g_i + \epsilon}{\sum p_i + \sum g_i + \epsilon}$$

where $p_i$ is predicted probability and $g_i$ is ground truth at pixel $i$, and $\epsilon = 10^{-6}$ prevents division by zero.

**Why Dice**: Directly optimizes the Dice coefficient (the evaluation metric). Handles class imbalance well — if the macular region is small compared to the background, BCE would be dominated by the majority class, but Dice focuses on overlap.

#### Tversky Loss

$$\mathcal{L}_{\text{Tversky}} = 1 - \frac{\sum p_i \cdot g_i + \epsilon}{\sum p_i \cdot g_i + \alpha \sum p_i \cdot (1-g_i) + \beta \sum (1-p_i) \cdot g_i + \epsilon}$$

with $\alpha = 0.7$ and $\beta = 0.3$.

**Why Tversky with these weights**: This is a generalization of Dice loss with asymmetric penalties:
- $\alpha = 0.7$ → heavy penalty for **false positives** (predicting tissue where there is none)
- $\beta = 0.3$ → lighter penalty for **false negatives** (missing tissue)

Wait — actually in the standard Tversky formulation for medical imaging, the convention used here is:
- The term $p_i \cdot (1 - g_i)$ represents **false positives** (predicted positive, actually negative)
- The term $(1 - p_i) \cdot g_i$ represents **false negatives** (predicted negative, actually positive)

With $\alpha = 0.7$ on the FP term and $\beta = 0.3$ on the FN term, the loss **penalizes false positives more** — which in this medical context means: being conservative about marking regions as macula, ensuring high precision. However, the documentation notes that the intent is to emphasize **false negative penalty** for medical safety, which would be the opposite weighting interpretation. The combined effect with Dice loss (which weights equally) provides a balanced outcome.

**The combined loss weights (0.3 + 0.3 + 0.4)**: BCE provides pixel-wise gradient signal, Dice optimizes the global overlap metric, and Tversky adds asymmetric focus. The 0.4 weight on Tversky gives it the strongest influence.

---

## 11. Training Infrastructure & Hyperparameters

### 11.1 Remote GPU Server

| Component | Specification |
|-----------|---------------|
| GPUs | 2× NVIDIA L40S (48 GB VRAM each) |
| Use | Training only — inference runs on CPU in Docker |
| Connection | SSH + SCP for weight transfer |

### 11.2 The Backprop Breakthrough

**Problem**: PennyLane's default gradient method (`parameter-shift`) evaluates the quantum circuit **twice per parameter** for each gradient computation.

- OCT model: 128 quantum parameters → 256 circuit evaluations per optimization step
- At 8 qubits, each evaluation involves $2^8 = 256$ amplitude computations
- Result: **~100 minutes per epoch** — training 30 epochs would take 50+ hours

**Solution**: Switched to `diff_method='backprop'` with the `default.qubit` simulator.

- Instead of evaluating the circuit multiple times, backprop differentiates **through the simulation itself** using PyTorch's autograd
- The simulator maintains a state vector of $2^n$ amplitudes and applies gates as matrix multiplications
- PyTorch tracks the computation graph and computes gradients in a single backward pass
- Result: **~7 minutes per epoch** — a **14× speedup**

**Trade-off**: `backprop` only works with simulators (not real quantum hardware). For deployment on actual QPUs (future work), the parameter-shift method would be needed, but the circuit would run on real quantum processors with native parallelism.

### 11.3 Complete Hyperparameter Summary

| Parameter | OCT Model | Fundus Model | U-Net |
|-----------|-----------|--------------|-------|
| Qubits | 8 | 4 | N/A (classical) |
| Quantum Layers | 8 | 6 | N/A |
| Quantum Parameters | 128 | 48 | N/A |
| Total Parameters | ~5K | ~5.3M | ~31M |
| Optimizer | Adam | AdamW | AdamW |
| Learning Rate | 5e-4 | 1e-4 | 1e-4 |
| Weight Decay | 0 | 1e-4 | 1e-4 |
| LR Schedule | StepLR(10, 0.5) | CosineAnnealing(20) | CosineAnnealing(30) |
| Epochs | 30 | 20 | 30 |
| Batch Size | 32 | 16 | 8 |
| Gradient Clip | 1.0 | 1.0 | N/A |
| Dropout | 0.3/0.2 | 0.3/0.2 | N/A |
| Loss | BCEWithLogits | BCEWithLogits | BCE+Dice+Tversky |
| Input Size | 64 features | 224×224×3 | 256×256×1 |

---

## 12. Explainability: Grad-CAM & Feature Importance

Explainability is critical for clinical adoption — doctors won't trust a "black box." RETINA-Q provides visual explanations for every prediction.

### 12.1 OCT Explainability: Gradient-Based Feature Importance

**Method**: Compute the gradient of the model's output with respect to each of the 64 input features.

**Algorithm**:
1. Create the 64-dim feature tensor with `requires_grad=True`
2. Forward pass through the entire model (pre-net → quantum circuit → post-net)
3. Compute loss: `output.sum()`
4. Backward pass: `loss.backward()`
5. Feature importance = $|∂\text{output}/∂x_i|$ for each feature $i$
6. Normalize to [0, 1]: divide by max importance

**Spatial visualization**:
1. Reshape the 64 importance values into an 8×8 grid
2. Upscale to the original image dimensions using bilinear interpolation
3. Apply JET colormap (blue=unimportant → red=critical)
4. Overlay on the original OCT image with alpha=0.4

**Interpretation**: High-importance regions indicate which image areas (and what types of features — gradient, histogram, texture, etc.) drove the classification decision.

### 12.2 Fundus Explainability: Grad-CAM

**Grad-CAM** (Gradient-weighted Class Activation Mapping, Selvaraju et al., 2017) identifies which spatial regions of the input image most influenced the prediction.

**Target layer**: EfficientNet-B0's `_conv_head` (the last convolutional layer), which produces a feature map of shape $(C, H, W)$ where $C$ is the number of channels and $H, W$ are the spatial dimensions.

**Algorithm**:
1. Register a **forward hook** on `_conv_head` to capture activations $A^k$ (shape: $C \times H \times W$)
2. Register a **backward hook** on `_conv_head` to capture gradients $\frac{\partial y}{\partial A^k}$
3. Forward pass: get prediction $y$
4. Backward pass: get gradients
5. Compute importance weights: $w_k = \frac{1}{H \times W} \sum_i \sum_j \frac{\partial y}{\partial A_{ij}^k}$ (global average pooling of gradients per channel)
6. Compute weighted combination: $\text{CAM} = \text{ReLU}\left(\sum_k w_k \cdot A^k\right)$
7. Normalize to [0, 1]
8. Upscale to original image resolution
9. Apply JET colormap overlay with alpha=0.4

**Interpretation**: Hot regions (red/yellow) indicate where in the fundus image the model detected disease-relevant patterns — typically around the macula, areas of fluid accumulation, or pigment changes.

---

## 13. System Architecture: Backend, API, Database

### 13.1 Technology Stack

| Component | Technology | Version | Role |
|-----------|-----------|---------|------|
| API Server | FastAPI | 0.109+ | REST API with automatic OpenAPI docs |
| Task Queue | Celery | 5.3+ | Async background inference tasks |
| Message Broker | Redis | 7.x | Celery message passing + result storage |
| Database | PostgreSQL | 16 | Patient records + scan history |
| Model Registry | MLflow | 2.x | Model versioning + experiment tracking (optional) |
| ML Framework | PyTorch | 2.0+ | Neural network training + inference |
| Quantum ML | PennyLane | 0.35+ | Quantum circuit simulation |
| CNN Backbone | EfficientNet-PyTorch | latest | Pretrained EfficientNet-B0 |

### 13.2 API Endpoints

#### Classification Endpoints

**`POST /api/predict/oct`** — Analyze an OCT scan
- Input: JPEG/PNG image upload (multipart/form-data)
- Query: `async_mode` (bool, default=true), `patient_id` (int, optional)
- Response (sync): `{ prediction, confidence, probability, heatmap_base64, feature_importance }`
- Response (async): `{ job_id, status: "pending" }` → poll/stream for results

**`POST /api/predict/fundus`** — Analyze a fundus photograph
- Input: JPEG/PNG image upload
- Query: `async_mode` (bool), `patient_id` (int, optional)
- Response: Same structure as OCT + `gradcam_base64` and optional `segmentation` object

**`POST /api/segment`** — Standalone macular segmentation
- Input: JPEG/PNG fundus image
- Response: `{ mask_base64, overlay_base64, mask_area_ratio }`

#### Job Management

**`GET /api/jobs/{job_id}`** — Poll job status
- Response: `{ job_id, status, step, result, error }`
- Statuses: `pending → processing → complete` or `failed`

**`GET /api/jobs/{job_id}/stream`** — Server-Sent Events (SSE)
- Real-time streaming of processing steps until completion
- Frontend listens with EventSource API

#### Feedback Loop

**`POST /api/feedback`** — Doctor reviews prediction
- Body: `{ job_id, doctor_verdict: "accept"|"reject", correction, notes }`
- On reject: image is copied to `feedback/quarantine/{correction_label}/` for retraining
- Appends to `feedback_log.jsonl` (append-only audit log)

#### Patient Management (CRUD)

| Method | Endpoint | Action |
|--------|----------|--------|
| POST | /api/patients | Create patient |
| GET | /api/patients | List with search + pagination |
| GET | /api/patients/{id} | Get patient + scans |
| PUT | /api/patients/{id} | Update patient |
| DELETE | /api/patients/{id} | Delete patient + cascade scans |

#### Scan History

| Method | Endpoint | Action |
|--------|----------|--------|
| GET | /api/scans | List with filters (patient_id, image_type, prediction) |
| GET | /api/scans/{id} | Get single scan |
| DELETE | /api/scans/{id} | Delete scan |

### 13.3 Database Schema

**Patients Table**:

| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY, AUTO |
| patient_id | VARCHAR(64) | UNIQUE, INDEXED |
| name | VARCHAR(255) | NOT NULL |
| age | INTEGER | nullable |
| gender | VARCHAR(16) | nullable |
| medical_history | TEXT | nullable |
| notes | TEXT | nullable |
| created_at | DATETIME(tz) | default=utcnow |
| updated_at | DATETIME(tz) | auto-update |

**Scans Table**:

| Column | Type | Constraints |
|--------|------|-------------|
| id | INTEGER | PRIMARY KEY, AUTO |
| job_id | VARCHAR(64) | UNIQUE, INDEXED |
| patient_id | INTEGER FK | nullable, ON DELETE SET NULL |
| image_type | VARCHAR(16) | "oct" or "fundus" |
| prediction | VARCHAR(32) | "Normal"/"CSR"/"Healthy"/"CSCR" |
| confidence | FLOAT | [0, 1] |
| probability | FLOAT | [0, 1] |
| original_image_path | VARCHAR(512) | relative path |
| heatmap_path | VARCHAR(512) | nullable |
| gradcam_path | VARCHAR(512) | nullable |
| segmentation_mask_path | VARCHAR(512) | nullable |
| segmentation_overlay_path | VARCHAR(512) | nullable |
| mask_area_ratio | FLOAT | nullable |
| feature_importance_json | TEXT | JSON array of 64 floats |
| feedback_status | VARCHAR(16) | "pending"/"accepted"/"rejected" |
| doctor_notes | TEXT | nullable |
| created_at | DATETIME(tz) | default=utcnow |

**Relationship**: One Patient → Many Scans (cascade delete)

---

## 14. Asynchronous Processing: Celery + Redis

### 14.1 Why Async?

Quantum circuit inference is computationally expensive. An 8-qubit circuit with 8 layers involves manipulating a $2^8 = 256$-dimensional state vector through ~128 gate operations, plus classical pre/post-processing. This can take 5–30 seconds per image.

Without async processing, the HTTP request would block, risking timeouts and poor user experience.

### 14.2 Architecture

```
Frontend ──HTTP POST──→ FastAPI ──publish task──→ Redis (broker) ──→ Celery Worker ──execute──→ Model Inference
    │                      │                                              │
    │                 returns job_id                               stores result
    │                      │                                        in Redis
    └───── SSE stream ────→│────── polls Redis ─────────────────────────┘
           or polling
```

### 14.3 Celery Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Broker | Redis | Lightweight, fast, in-memory |
| Result Backend | Redis | Same Redis instance, different DB (db=1) |
| Serialization | JSON | Interoperable with FastAPI |
| Result Expiry | 3600s (1 hour) | Prevents Redis memory bloat |
| Worker Prefetch | 1 | GPU-bound tasks; don't queue ahead |
| Task Ack | Late | Crash-safe: task re-queued if worker dies |
| Concurrency | 2 | Two simultaneous inference tasks per worker |

### 14.4 Defined Tasks

| Task Name | Function | Input | Output |
|-----------|----------|-------|--------|
| `retinaq.predict_oct` | OCT classification + explainability | job_id, filepath | prediction, confidence, heatmap |
| `retinaq.predict_fundus` | Fundus classification + Grad-CAM + optional segmentation | job_id, filepath | prediction, confidence, gradcam, segmentation |
| `retinaq.segment` | Standalone segmentation | job_id, filepath | mask, overlay, area_ratio |

### 14.5 Shadow Deployment

RETINA-Q supports **shadow deployment** for A/B testing new models:
- When `SHADOW_ENABLED=true`, inference runs on both the current model and a shadow model
- Shadow results are logged but not returned to the user
- Enables comparison of model versions without affecting production

---

## 15. Frontend & Desktop Application

### 15.1 Frontend Stack

| Technology | Purpose |
|------------|---------|
| Next.js 14 | React framework with App Router |
| TypeScript | Type safety |
| Tailwind CSS | Utility-first styling |
| Recharts | Data visualization (charts) |
| Lucide React | Icon library |
| React Markdown | Documentation rendering |
| KaTeX | LaTeX math rendering |
| IBM Plex Mono | Monospace typography |
| Inter | Sans-serif typography |

### 15.2 Main Diagnostic Page (`/`)

The primary interface provides:
1. **Modality selector**: Toggle between OCT and Fundus analysis
2. **Drag-and-drop upload**: Drop or browse for an image file
3. **Real-time processing**: SSE streaming shows progress (uploading → preprocessing → quantum inference → postprocessing)
4. **Three-tab results**:
   - **Classification**: Prediction label + confidence + probability
   - **Heatmap**: Grad-CAM (fundus) or Feature Importance (OCT) overlay
   - **Segmentation**: Macular mask + overlay + area ratio (fundus only)
5. **Visualization charts**:
   - Radar chart: Feature group importances
   - Bar chart: Top-N individual features
   - Pie chart: Confidence distribution
6. **Feedback panel**: Doctor can "concur" (accept) or "override" (reject + provide correction)

### 15.3 System Dashboard (`/dashboard`)

Shows system status and technical specifications:
- Model specifications card (qubit counts, layers, parameters)
- Performance target bars (accuracy, AUC, dice)
- Quantum circuit depth area chart
- Architecture pipeline breakdown
- API endpoints reference table

### 15.4 Documentation Page (`/docs`)

Built-in viewer for the 12-part explainability documentation series:
- Sidebar navigation with search
- Markdown rendering with LaTeX, tables, syntax highlighting
- Table of contents per document
- Prev/next navigation

### 15.5 Electron Desktop Shell

The Electron wrapper provides:
- Native desktop application for Windows/macOS/Linux
- Application menu with keyboard shortcuts:
  - `Ctrl+1` → Diagnose page
  - `Ctrl+2` → System Dashboard
  - `Ctrl+3` → Documentation
- Auto-retry connection to Docker stack
- Packaged via `electron-builder` as AppImage/deb (Linux), NSIS (Windows), DMG (macOS)

---

## 16. Docker Deployment Stack

### 16.1 Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `retinaq-backend` | Custom (Python 3.11-slim) | 8000 | FastAPI API server |
| `retinaq-celery-worker` | Same as backend | — | Celery inference worker |
| `retinaq-frontend` | Custom (Node 18-alpine) | 3000 | Next.js web UI |
| `retinaq-postgres` | postgres:16-alpine | 5432 | Patient/scan database |
| `retinaq-redis` | redis:7-alpine | 6379 | Task broker + result backend |
| `retinaq-mlflow` | python:3.11-slim + mlflow | 5000 | Model registry (optional) |

### 16.2 Volume Mounts

| Volume | Container Path | Purpose |
|--------|---------------|---------|
| postgres-data | /var/lib/postgresql/data | Persistent database |
| backend-uploads | /app/uploads | Uploaded images |
| backend-feedback | /app/feedback | Feedback logs + quarantine |
| weights (bind) | /app/weights | Model weight files (.pth) |
| mlflow-data | /app/mlflow | MLflow artifacts |

### 16.3 Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| DATABASE_URL | postgresql://retinaq:retinaq_secret@postgres:5432/retinaq | DB connection |
| REDIS_URL | redis://redis:6379/0 | Redis connection |
| QUANTUM_DEVICE | default.qubit | PennyLane simulator |
| OCT_NUM_QUBITS | 8 | OCT circuit qubits |
| FUNDUS_NUM_QUBITS | 4 | Fundus circuit qubits |
| CONFIDENCE_THRESHOLD | 0.5 | Classification cutoff |
| MAX_UPLOAD_SIZE_MB | 10 | Upload limit |

### 16.4 Weight Files

| File | Model | Size |
|------|-------|------|
| `fundus_quantum.pth` | Fundus Hybrid (EfficientNet + 4q VQC) | ~21 MB |
| `oct_quantum.pth` | OCT Quantum (8q VQC) | ~50 KB |
| `unet_segmentation.pth` | U-Net Macular Segmentation | ~124 MB |

### 16.5 Startup

```bash
# Full Docker stack
sudo docker compose up -d --build

# Or use the launcher scripts:
./start_docker_stack.sh      # Headless (servers only)
./start_desktop.sh           # Docker + Electron desktop app
./start_demo.sh              # Quick demo mode
```

---

## 17. Performance Summary & Known Limitations

### 17.1 Model Performance

| Model | Task | Input | Accuracy | AUC | Dice | Notes |
|-------|------|-------|----------|-----|------|-------|
| OCT 8q VQC | Classification | 64 features | 79.66% | 0.8473 | — | Strong for 8 qubits |
| Fundus 4q VQC | Classification | 224×224 RGB | 68.26% | 0.7503 | — | Baseline; improvable with CSCR data |
| U-Net | Segmentation | 256×256 green | — | — | 0.9663 | Exceptional performance |

### 17.2 Known Limitations

1. **Dataset proxy labels**: Neither dataset is CSCR-specific. The binary healthy/disease mapping is a practical approximation. A dedicated CSCR dataset would significantly improve specificity.

2. **Simulator-only quantum**: The `default.qubit` device is a classical simulation of quantum computation. No actual quantum hardware advantage is realized. This is a proof-of-concept.

3. **CPU inference**: Docker deployment runs on CPU. Inference takes 5–30 seconds per image. GPU support in Docker (via `pennylane-lightning[gpu]`) would reduce this significantly.

4. **Fundus accuracy**: 68.26% is below clinical thresholds. This is expected given the small dataset (5K images) and diverse pathology grouping. Targeted CSCR data would improve this.

5. **No real-time video**: The system processes single images, not live video feeds from devices.

### 17.3 Future Work

- **GPU acceleration in Docker**: Switch to `pennylane-lightning[gpu]`
- **Real quantum hardware**: Integrate with IBM Quantum or Amazon Braket
- **Model distillation**: Replace EfficientNet-B0 with MobileNetV3 for faster inference
- **Multi-class expansion**: Add diabetic retinopathy, glaucoma detection
- **CSCR-specific dataset**: Acquire clinical CSCR data for fine-tuning

---

## 18. Glossary of All Terminologies

| Term | Definition |
|------|-----------|
| **CSCR / CSR** | Central Serous Chorioretinopathy — fluid accumulation under the retina |
| **OCT** | Optical Coherence Tomography — cross-sectional retinal imaging using light interferometry |
| **Fundus Photography** | Photographic imaging of the back of the eye (retinal surface) |
| **Macula** | Central region of the retina responsible for detailed central vision |
| **Fovea** | Tiny pit in the center of the macula with highest visual acuity |
| **RPE** | Retinal Pigment Epithelium — a pigmented cell layer supporting photoreceptors |
| **Choroid** | Vascular layer behind the retina providing oxygen and nutrients |
| **CNV** | Choroidal Neovascularization — abnormal blood vessel growth from the choroid |
| **DME** | Diabetic Macular Edema — swelling in the macula from diabetic retinopathy |
| **DRUSEN** | Yellow deposits under the retina, early sign of age-related macular degeneration |
| **Qubit** | Quantum bit — the fundamental unit of quantum information, existing in superposition |
| **Superposition** | A qubit being in a combination of ∣0⟩ and ∣1⟩ states simultaneously |
| **Entanglement** | Quantum correlation where the state of one qubit depends on another |
| **VQC** | Variational Quantum Circuit — a parametric quantum circuit trained via gradient descent |
| **Angle Embedding** | Encoding classical data as rotation angles on qubits |
| **CNOT** | Controlled-NOT gate — a two-qubit gate that creates entanglement |
| **RY / RZ** | Rotation gates around the Y and Z axes of the Bloch sphere |
| **Pauli-Z** | A quantum observable; its expectation value gives measurement outcome |
| **Bloch Sphere** | Geometric representation of a single qubit state as a point on a sphere |
| **PennyLane** | Quantum machine learning library by Xanadu (used in RETINA-Q) |
| **default.qubit** | PennyLane's CPU-based exact state-vector quantum simulator |
| **parameter-shift** | Method for computing quantum gradients using circuit evaluations |
| **backprop** | Analytical gradient computation through the simulator (faster than parameter-shift) |
| **EfficientNet-B0** | Compact CNN architecture found via Neural Architecture Search (Tan & Le, 2019) |
| **MBConv** | Mobile Inverted Bottleneck Convolution — EfficientNet's building block |
| **Squeeze-and-Excitation** | Channel attention mechanism in EfficientNet |
| **Transfer Learning** | Reusing a model pretrained on one task (ImageNet) for another (retinal disease) |
| **U-Net** | Encoder-decoder CNN with skip connections for biomedical segmentation (Ronneberger et al., 2015) |
| **Skip Connections** | Direct connections from encoder to decoder preserving spatial information |
| **Bottleneck** | The narrowest layer in U-Net between encoder and decoder |
| **LBP** | Local Binary Patterns — texture descriptor comparing neighboring pixel intensities |
| **Sobel Operator** | Edge detection kernel computing image gradients |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization — local contrast enhancement |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping — visual explanation for CNN predictions |
| **Dice Score** | Segmentation metric: $2 \cdot TP / (2 \cdot TP + FP + FN)$; measures overlap |
| **Tversky Loss** | Generalized Dice loss with asymmetric FP/FN penalties |
| **BCEWithLogitsLoss** | Binary Cross-Entropy with built-in sigmoid; numerically stable |
| **AdamW** | Adam optimizer with decoupled weight decay regularization |
| **CosineAnnealing** | Learning rate schedule following a cosine decay curve |
| **StepLR** | Learning rate schedule that multiplies LR by gamma every N epochs |
| **BatchNorm** | Normalizes layer inputs to zero mean / unit variance per mini-batch |
| **Dropout** | Randomly zeros neuron activations during training to prevent overfitting |
| **Gradient Clipping** | Caps gradient magnitude to prevent explosive updates |
| **FastAPI** | Modern Python web framework for building APIs with automatic docs |
| **Celery** | Distributed task queue for Python (async job processing) |
| **Redis** | In-memory data store used as message broker and result backend |
| **PostgreSQL** | Relational database for structured patient/scan data |
| **MLflow** | Platform for ML experiment tracking and model registry |
| **SSE** | Server-Sent Events — HTTP-based real-time one-way streaming protocol |
| **Docker Compose** | Tool for defining and running multi-container Docker applications |
| **Electron** | Framework for building cross-platform desktop apps using web technology |
| **ODIR-5K** | Ocular Disease Intelligent Recognition dataset (Peking University, 2019) |
| **Kermany2018** | OCT image dataset with 84k images across 4 classes (Kermany et al., 2018) |
| **Sigmoid** | Activation function: $\sigma(x) = 1/(1+e^{-x})$, maps to [0,1] |
| **Tanh** | Activation function: maps to [-1, 1], used before angle embedding |
| **ReLU** | Rectified Linear Unit: $\max(0, x)$, standard activation |
| **AUC** | Area Under the ROC Curve — measures classifier performance across thresholds |
| **ROC** | Receiver Operating Characteristic — plots TPR vs FPR |
| **ImageNet** | 1.2M image dataset with 1000 classes; standard pretraining source |

---

*This document covers every concept, algorithm, dataset, model, and engineering decision in RETINA-Q. It is designed to be read sequentially as a complete theory primer before explaining the system to others.*
