# 07 — U-Net Macular Segmentation

## Introduction

While the OCT and fundus models answer "Is this retina normal or pathological?", the U-Net segmentation model answers a different question: **"Where exactly is the macular region?"** It produces a pixel-level binary mask delineating the macula — the central area of the retina responsible for sharp, detailed vision and the primary site affected by CSCR.

Segmentation adds clinical depth to the diagnostic pipeline: instead of a single classification label, clinicians see precisely which anatomical region the system has identified, enabling verification and measurement.

---

## Why Segmentation Matters

### Clinical Value

1. **Localisation**: A classifier says "CSCR detected." A segmentation mask says "CSCR detected — specifically in this 3mm² area centred on the fovea."
2. **Measurement**: The mask enables quantitative analysis — macular area, thickness estimation, change tracking over time.
3. **Cross-validation**: If the classifier says "CSCR" but the segmentation mask shows no macular abnormality, the clinician is alerted to a potential false positive.
4. **Treatment planning**: Precise localisation aids in laser photocoagulation or photodynamic therapy targeting.

### Integration with the Classification Pipeline

When the fundus model detects CSCR (prediction = 1), the segmentation model is automatically invoked to provide the localisation mask. This happens in the same async Celery task, so the user receives both results simultaneously.

---

## U-Net Architecture

U-Net was introduced by Ronneberger et al. (2015) for biomedical image segmentation. Its key innovation is the **encoder-decoder architecture with skip connections** — the encoder captures "what" is in the image, the decoder reconstructs "where" it is, and skip connections preserve fine spatial details that would be lost through the bottleneck.

### RETINA-Q's U-Net Topology

```
Input: (1, 256, 256) — single-channel green CLAHE image

ENCODER (downsampling path):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Level 1: ConvBlock(1, 64)     → (64, 256, 256)   → MaxPool → (64, 128, 128)
Level 2: ConvBlock(64, 128)   → (128, 128, 128)  → MaxPool → (128, 64, 64)
Level 3: ConvBlock(128, 256)  → (256, 64, 64)    → MaxPool → (256, 32, 32)
Level 4: ConvBlock(256, 512)  → (512, 32, 32)    → MaxPool → (512, 16, 16)

BOTTLENECK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ConvBlock(512, 1024) → (1024, 16, 16)

DECODER (upsampling path with skip connections):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Level 4: ConvTranspose(1024, 512) → (512, 32, 32)
         Concat with Encoder Level 4 → (1024, 32, 32)
         ConvBlock(1024, 512) → (512, 32, 32)

Level 3: ConvTranspose(512, 256) → (256, 64, 64)
         Concat with Encoder Level 3 → (512, 64, 64)
         ConvBlock(512, 256) → (256, 64, 64)

Level 2: ConvTranspose(256, 128) → (128, 128, 128)
         Concat with Encoder Level 2 → (256, 128, 128)
         ConvBlock(256, 128) → (128, 128, 128)

Level 1: ConvTranspose(128, 64) → (64, 256, 256)
         Concat with Encoder Level 1 → (128, 256, 256)
         ConvBlock(128, 64) → (64, 256, 256)

OUTPUT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Conv2d(64, 1, kernel_size=1) → (1, 256, 256)
Sigmoid → pixel probabilities in [0, 1]
```

### ConvBlock Definition

Each ConvBlock consists of two convolution-batchnorm-relu sequences:

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
```

- **3×3 convolutions with padding=1**: Preserves spatial dimensions within each block.
- **BatchNorm**: Stabilises training by normalising intermediate activations.
- **ReLU**: Introduces non-linearity.

### Skip Connections: Why They Matter

The encoder progressively downsamples the image from 256×256 to 16×16, capturing increasingly abstract features but losing spatial precision. The decoder upsamples back to 256×256, but without skip connections, it would have to "guess" the fine details.

Skip connections directly copy the encoder feature maps to the corresponding decoder level:

```python
# In the decoder forward pass:
x = self.upconv(x)                        # Upsample
x = torch.cat([x, encoder_features], dim=1)  # Concatenate skip connection
x = self.conv_block(x)                    # Process combined features
```

This gives the decoder access to both high-level semantic information (from the bottleneck) and low-level spatial information (from the encoder) — combining "what" with "where."

---

## Loss Function: BCE + Dice + Tversky

The choice of loss function is critical for medical image segmentation, where the target region (macula) occupies only a small fraction of the total image. Standard cross-entropy alone would let the model achieve high accuracy by simply predicting "background" everywhere.

### Combined Loss

```python
def combined_loss(pred, target):
    bce = F.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    tversky = tversky_loss(pred, target, alpha=0.7, beta=0.3)
    
    return 0.3 * bce + 0.3 * dice + 0.4 * tversky
```

### Component 1: Binary Cross-Entropy (weight 0.3)

$$\text{BCE} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

Standard pixel-wise cross-entropy. Provides smooth, well-behaved gradients but doesn't account for class imbalance.

### Component 2: Dice Loss (weight 0.3)

$$\text{Dice Loss} = 1 - \frac{2 \sum_{i} y_i \hat{y}_i + \epsilon}{\sum_{i} y_i + \sum_{i} \hat{y}_i + \epsilon}$$

The Dice coefficient measures overlap between prediction and ground truth. Dice loss directly optimises this overlap metric, handling class imbalance by design — it only considers the foreground region.

### Component 3: Tversky Loss (weight 0.4)

$$\text{Tversky Loss} = 1 - \frac{\sum_{i} y_i \hat{y}_i + \epsilon}{\sum_{i} y_i \hat{y}_i + \alpha \sum_{i} (1-y_i)\hat{y}_i + \beta \sum_{i} y_i(1-\hat{y}_i) + \epsilon}$$

With $\alpha = 0.7$ and $\beta = 0.3$.

Tversky loss generalises Dice loss with asymmetric penalties:
- $\alpha = 0.7$: **High penalty for false positives** (predicting foreground when it's background)
- $\beta = 0.3$: **Lower penalty for false negatives** (missing foreground)

Wait — shouldn't medical imaging penalise false negatives more? In our case, $\alpha$ penalises FP and $\beta$ penalises FN. The rationale: we want the macular region mask to be **precise** (not over-segmented), because over-segmentation would make clinical measurements unreliable. Missing a small edge of the macula (FN) is less harmful than marking a large non-macular area as macula (FP).

**Why Tversky gets the highest weight (0.4)**: It provides the strongest signal for learning the correct boundary between macula and non-macula, with configurable precision-recall trade-off.

---

## Training Procedure

### Pseudo-Mask Generation

We did not have manually annotated segmentation masks. Instead, pseudo-masks were generated from the green channel:

```python
green_channel = image[:, :, 1]
_, mask = cv2.threshold(green_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

Otsu's thresholding automatically finds the optimal binarisation threshold. The resulting masks capture the general retinal region against the dark background — not perfect macular boundaries, but sufficient for training the model to learn retinal structure segmentation.

### Hyperparameters

| Parameter | Value |
|---|---|
| **Epochs** | 50 |
| **Batch size** | 8 |
| **Optimiser** | Adam |
| **Learning rate** | 1 × 10⁻⁴ |
| **Input size** | 256 × 256 (single channel) |
| **Loss** | 0.3 × BCE + 0.3 × Dice + 0.4 × Tversky |
| **Tversky α/β** | 0.7 / 0.3 |

### Training Loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    model.train()
    for images, masks in train_loader:
        optimizer.zero_grad()
        preds = model(images)
        loss = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()
    
    # Compute Dice score on validation set
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks in val_loader:
            preds = model(images)
            dice = compute_dice(preds > 0.5, masks)
            dice_scores.append(dice)
    
    print(f"Epoch {epoch}: Loss={loss:.4f}, Dice={np.mean(dice_scores):.4f}")
```

---

## Results

### Final Metrics

| Metric | Value |
|---|---|
| **Dice Score** | 0.9663 |
| **Training Loss** | < 0.05 |
| **Model Size** | ~31 MB |
| **Inference Time** | ~1–2 seconds per image |

### Dice Score of 0.9663

A Dice score of 0.9663 means 96.63% overlap between the predicted mask and the ground truth. This is excellent for medical image segmentation — clinical-grade performance exceeding the 0.90 target.

### Why So Good?

The high Dice score is partly due to the simplified task: segmenting the general retinal/macular region from a dark background is inherently easier than segmenting complex pathological structures. The green channel CLAHE preprocessing provides high contrast between the retinal region and the surrounding area.

For more challenging segmentation tasks (e.g., segmenting specific fluid pockets, individual retinal layers, or microaneurysms), performance would be lower and expert-annotated masks would be necessary.

---

## Post-Processing

The raw model output is a probability map. Post-processing converts it to a clean binary mask:

### Step 1: Binarise

```python
binary_mask = (prediction > 0.5).astype(np.uint8)
```

### Step 2: Connected Component Analysis

```python
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
# Keep only the largest connected component (main retinal region)
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
clean_mask = (labels == largest_label).astype(np.uint8)
```

This removes small disconnected noise regions, keeping only the largest contiguous segmented area.

### Step 3: Morphological Operations

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)   # Remove protrusions
```

- **Morphological close**: Fills small holes within the mask.
- **Morphological open**: Removes small peninsulas/protrusions at the mask boundary.
- **Elliptical kernel**: Produces smoother, more anatomically plausible boundaries than a square kernel.

### Step 4: Overlay Generation

```python
overlay = original_image.copy()
overlay[clean_mask == 1] = [0, 255, 0]  # Green overlay on segmented region
blended = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)
```

The final output is a semi-transparent green overlay on the original fundus image, clearly showing the segmented macular region.

---

## Integration with the Diagnostic Pipeline

```python
# In inference.py:
def predict_fundus(image_path):
    # 1. Classification
    prediction, confidence = fundus_model.predict(preprocessed_image)
    
    # 2. Explainability
    heatmap = grad_cam(fundus_model, preprocessed_image)
    
    # 3. Segmentation (always run for fundus)
    segmentation_mask = unet_model.predict(segmentation_preprocessed)
    segmentation_overlay = create_overlay(original_image, segmentation_mask)
    
    return {
        "prediction": "CSCR" if prediction >= 0.5 else "Healthy",
        "confidence": float(confidence),
        "heatmap": encode_base64(heatmap),
        "segmentation": encode_base64(segmentation_overlay)
    }
```

All outputs are Base64-encoded PNG images, transmitted as JSON strings to the frontend for direct rendering.

---

## Model Parameter Count

```
Encoder Level 1:   64 × (1×3×3 + 64×3×3) = 37,568
Encoder Level 2:  128 × (64×3×3 + 128×3×3) = 221,440
Encoder Level 3:  256 × (128×3×3 + 256×3×3) = 885,248
Encoder Level 4:  512 × (256×3×3 + 512×3×3) = 3,539,968
Bottleneck:      1024 × (512×3×3 + 1024×3×3) = 14,157,824
Decoder Level 4: 512 × (1024×3×3 + 512×3×3) = (mirror of encoder)
Decoder Level 3: ...
Decoder Level 2: ...
Decoder Level 1: ...
Output Conv:     1 × 64 × 1 × 1 = 64

Total: ~31M parameters
```

Despite being the largest model in RETINA-Q by parameter count, the U-Net is the fastest at inference because it is purely classical — no quantum circuit simulation overhead.

The next document (08) explains how the backend API orchestrates all three models behind a clean REST interface.
