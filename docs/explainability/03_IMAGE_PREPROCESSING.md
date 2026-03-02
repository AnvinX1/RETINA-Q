# 03 — Image Preprocessing Pipeline

## Introduction

Raw retinal images cannot be fed directly into a neural network. They vary in resolution, contrast, colour balance, and noise levels. The preprocessing pipeline transforms these raw clinical images into standardised tensors that the models can consume — and does so in a way that preserves diagnostically relevant information while suppressing irrelevant variation.

This document details every preprocessing step for both the OCT and fundus pipelines.

---

## OCT Image Preprocessing

OCT images are greyscale cross-sections of the retina. The preprocessing pipeline must handle variable resolutions, contrast differences between imaging devices, and the fact that the model consumes a **64-dimensional feature vector** rather than raw pixels.

### Step 1: Load and Convert to Greyscale

```python
from PIL import Image

image = Image.open(image_path).convert("L")  # Force greyscale
```

Even though OCT images are inherently greyscale, some may be saved as RGB. The `.convert("L")` call normalises this.

### Step 2: Resize to Standard Dimensions

```python
image = image.resize((224, 224), Image.BILINEAR)
```

All images are resized to **224 × 224 pixels** — a standard input size that balances detail retention with computational efficiency. Bilinear interpolation is used to avoid aliasing artifacts.

### Step 3: Convert to NumPy Array and Normalise

```python
import numpy as np

img_array = np.array(image, dtype=np.float32) / 255.0
```

Pixel values are scaled from [0, 255] to [0.0, 1.0]. This normalisation ensures consistent gradient magnitudes during training.

### Step 4: Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)

```python
import cv2

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_uint8 = (img_array * 255).astype(np.uint8)
enhanced = clahe.apply(img_uint8)
enhanced = enhanced.astype(np.float32) / 255.0
```

**Why CLAHE?** OCT images from different devices and settings often have vastly different contrast profiles. Standard histogram equalisation can over-amplify noise, but CLAHE applies equalisation locally (in 8 × 8 tiles) with a clipping limit that prevents noise amplification. This reveals subtle structural details — like thin fluid layers or RPE irregularities — that may be invisible in low-contrast regions.

### Step 5: 64-Dimensional Feature Extraction

Rather than feeding raw pixels into the quantum circuit (which would require thousands of qubits), we extract a compact 64-dimensional feature vector from each preprocessed OCT image. This is a critical design decision that makes quantum processing tractable.

The feature extraction pipeline (`oct_feature_extractor.py`) computes four groups of features:

#### A. Gradient Magnitude Statistics (16 features)

```python
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(grad_x**2 + grad_y**2)
```

From the gradient magnitude map, we compute:
- Mean, std, max, min of overall gradient
- Mean gradient per quadrant (4 quadrants × 1 = 4 features)
- Horizontal and vertical gradient profiles (4 features each)

**Clinical relevance**: Retinal layer boundaries produce strong gradients. CSCR disrupts these boundaries — subretinal fluid creates blurred or absent gradient edges.

#### B. Histogram Distribution (16 features)

```python
hist = np.histogram(image, bins=16, range=(0.0, 1.0))[0]
hist = hist / hist.sum()  # Normalise to probability distribution
```

A 16-bin normalised histogram captures the overall intensity distribution.

**Clinical relevance**: Healthy OCTs have a characteristic bimodal distribution (bright layers, dark vitreous). Fluid accumulation shifts the distribution toward mid-range values.

#### C. Local Binary Pattern (LBP) Features (16 features)

```python
from skimage.feature import local_binary_pattern

lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
lbp_hist = np.histogram(lbp, bins=16, range=(0, 16))[0]
lbp_hist = lbp_hist / lbp_hist.sum()
```

LBP computes texture patterns by comparing each pixel to its neighbours (8 neighbours at radius 1). The resulting histogram captures micro-texture structure.

**Clinical relevance**: Normal retinal tissue has regular, layered texture patterns. Pathological regions (fluid, drusen, neovascularisation) disrupt these patterns, producing distinct LBP distributions.

#### D. Texture Variance & Statistical Moments (16 features)

```python
# Texture variance in sub-regions
for region in split_into_subregions(image, 4):
    features.append(np.var(region))
    features.append(np.mean(region))

# Statistical moments
features.extend([
    np.mean(image), np.std(image),
    skew(image.flatten()), kurtosis(image.flatten()),
    np.percentile(image, 25), np.percentile(image, 75),
    iqr(image.flatten()), np.median(image)
])
```

- **Texture variance per quadrant** (8 features): Captures spatial heterogeneity.
- **Statistical moments** (8 features): Skewness captures asymmetry, kurtosis captures tail weight, IQR captures central spread.

**Clinical relevance**: Fluid pockets create localised high-variance regions. Statistical moments summarise the overall distributional shift caused by pathology.

### Final Output

The 64 features are concatenated into a single vector:

```python
features = np.concatenate([gradient_features, histogram_features, lbp_features, texture_features])
# Shape: (64,)
```

This vector is then converted to a PyTorch tensor and passed to the OCT quantum model's pre-network.

---

## Fundus Image Preprocessing

Fundus images are full-colour RGB photographs. The preprocessing is simpler because the EfficientNet-B0 backbone handles feature extraction internally.

### Step 1: Load as RGB

```python
image = Image.open(image_path).convert("RGB")
```

### Step 2: Resize

```python
image = image.resize((224, 224), Image.BILINEAR)
```

224 × 224 matches EfficientNet-B0's expected input dimensions.

### Step 3: Convert to Tensor and Normalise

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),               # [0,255] → [0.0, 1.0], CHW format
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],      # ImageNet means
        std=[0.229, 0.224, 0.225]         # ImageNet stds
    )
])
tensor = transform(image)  # Shape: (3, 224, 224)
```

**Why ImageNet normalisation?** The EfficientNet-B0 backbone is pretrained on ImageNet. Matching the normalisation statistics of the pretraining data ensures the frozen convolutional layers produce meaningful feature maps from the start.

### Training-Time Augmentation

During training only, additional augmentations are applied to improve generalisation:

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

| Augmentation | Purpose |
|---|---|
| `RandomCrop(224)` | Slight positional variation, simulates off-centre captures |
| `RandomHorizontalFlip` | Fundus images can be left or right eye; flipping doubles effective data |
| `RandomRotation(15°)` | Simulates slight camera angle variation |
| `ColorJitter` | Accounts for different camera white balances and lighting conditions |

---

## Segmentation Preprocessing

The U-Net segmentation model has its own dedicated preprocessing:

### Step 1: Extract Green Channel

```python
img_array = np.array(image)  # RGB
green_channel = img_array[:, :, 1]  # Index 1 = Green
```

**Why only the green channel?** In fundus photography, the green channel provides the best contrast for retinal vasculature and macular structures. Red tends to be over-saturated (high blood content), blue tends to be noisy (short wavelength scatter). Green is the clinical standard for vessel segmentation and macular analysis.

### Step 2: Apply CLAHE

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(green_channel)
```

Same adaptive contrast enhancement as the OCT pipeline, applied to the green channel to reveal fine macular detail.

### Step 3: Resize to 256 × 256

```python
enhanced = cv2.resize(enhanced, (256, 256))
```

The U-Net uses 256 × 256 input (slightly larger than the classifiers) because segmentation benefits from finer spatial resolution — it needs to produce a pixel-accurate mask.

### Step 4: Normalise and Add Channel Dimension

```python
tensor = torch.FloatTensor(enhanced / 255.0).unsqueeze(0)  # Shape: (1, 256, 256)
```

Single-channel greyscale input at float32 precision.

### Pseudo-Mask Generation for Training

Since we did not have manually annotated segmentation masks, the training script generates pseudo-masks from the green channel:

```python
green = image_array[:, :, 1]
_, mask = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

Otsu's method automatically finds the optimal threshold to separate foreground (retinal structures) from background. These pseudo-masks are imperfect but sufficient for training the model to identify the general macular region — which is then refined through the model's learned representations.

---

## Preprocessing Summary Table

| Stage | OCT Pipeline | Fundus Pipeline | Segmentation Pipeline |
|---|---|---|---|
| **Input** | Greyscale JPEG | RGB JPEG | RGB JPEG |
| **Colour handling** | Convert to L | Keep RGB | Extract green channel |
| **Resize** | 224 × 224 | 224 × 224 | 256 × 256 |
| **Contrast enhancement** | CLAHE (clip=2.0) | ImageNet normalisation | CLAHE (clip=2.0) |
| **Feature extraction** | 64-dim handcrafted vector | EfficientNet-B0 (1280-dim) | None (pixel-level) |
| **Output shape** | `(64,)` tensor | `(3, 224, 224)` tensor | `(1, 256, 256)` tensor |
| **Augmentation** | None (features are rotation-invariant) | Flip, crop, rotate, colour jitter | None |

---

## Design Rationale: Why Handcrafted Features for OCT?

A natural question: why not use a pretrained CNN for OCT feature extraction, as we did for fundus? The answer ties directly to the quantum computing constraint.

The 8-qubit quantum circuit can only accept 8 input values (one per qubit). Even with the pre-network reducing 64 features to 8, we need those 64 features to be **maximally informative** about retinal pathology. Handcrafted features allow us to inject domain knowledge directly:

- **Gradients** encode layer boundary integrity (the primary OCT diagnostic signal)
- **Histograms** encode overall brightness distribution (fluid changes this)
- **LBP** encodes local texture (pathology disrupts regular patterns)
- **Moments** encode global statistical properties (skewness, kurtosis shift under pathology)

A CNN backbone would give us 1280+ generic features that would need aggressive compression to reach 8 qubits — losing domain-specific signal in the process. The handcrafted approach preserves ophthalmological knowledge through the entire pipeline.

The next document (04 — Quantum Computing Fundamentals) explains the quantum circuits that process these features.
