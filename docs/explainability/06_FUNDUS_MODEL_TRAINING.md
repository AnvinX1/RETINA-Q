# 06 — Fundus Hybrid Model: Architecture & Training

## Introduction

The fundus classification pipeline is RETINA-Q's hybrid quantum-classical model — combining a pretrained **EfficientNet-B0** CNN backbone with a **4-qubit quantum circuit**. Unlike the OCT model, which relies entirely on handcrafted features, the fundus model leverages transfer learning from ImageNet and uses the quantum layer as a high-expressivity bottleneck that entangles the classical features before final classification.

This document covers the architecture design, the dual-pathway design philosophy, the training procedure, and the results obtained.

---

## End-to-End Pipeline

```
Fundus Image (224×224, RGB)
        │
        ▼
┌─────────────────────────────────────────┐
│     EfficientNet-B0 Backbone            │
│  (Pretrained on ImageNet, frozen/unfrozen│
│   depending on training stage)           │
│  → 1,280-dimensional feature vector      │
└─────────────────────┬───────────────────┘
                      │ (1280,)
                      ▼
┌─────────────────────────────────────────┐
│     Reduction Network                    │
│  Linear(1280 → 64) → ReLU → Dropout(0.3)│
│  Linear(64 → 4) → Tanh                  │
└───────────┬─────────────────────────────┘
            │ (4,)  ← values in [-1, +1]
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
┌──────────┐  ┌───────────────────────┐
│ Classical │  │ 4-Qubit Quantum       │
│ Pathway   │  │ Circuit               │
│ (pass-    │  │ AngleEmbedding → 6    │
│  through) │  │ layers of RY/RZ +     │
│           │  │ ring CNOT → PauliZ    │
│ 4 values  │  │ → 4 measurements      │
└─────┬─────┘  └───────────┬───────────┘
      │ (4,)              │ (4,)
      └────────┬──────────┘
               │ concatenate
               ▼
            (8,)  ← combined feature vector
               │
               ▼
┌─────────────────────────────────────────┐
│     Classification Head (Classical)     │
│  Linear(8 → 32) → ReLU → BN            │
│  → Dropout(0.3)                         │
│  Linear(32 → 16) → ReLU                 │
│  Linear(16 → 1) → Sigmoid               │
└─────────────────────┬───────────────────┘
                      │ (1,)
                      ▼
               Healthy (< 0.5) or CSCR (≥ 0.5)
```

---

## Architecture Deep Dive

### EfficientNet-B0: The Feature Backbone

EfficientNet-B0 is a CNN architecture that uses compound scaling to balance depth, width, and resolution. It was chosen for several reasons:

1. **Strong baseline**: Achieves 77.1% top-1 accuracy on ImageNet with only 5.3M parameters.
2. **Efficient**: Much smaller than ResNet-50 (25.6M params) or VGG-16 (138M params) while achieving better accuracy.
3. **Transfer learning**: Features learned from ImageNet's 1.2M natural images transfer surprisingly well to medical imaging tasks.

```python
from efficientnet_pytorch import EfficientNet

self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
# Output: 1,280-dimensional feature vector after global average pooling
```

The backbone's final classification layer is removed. We use only the feature extractor portion, which outputs a 1,280-dimensional vector per input image.

### Reduction Network: 1280 → 4

The quantum circuit uses only 4 qubits, so we need aggressive dimensionality reduction:

```python
self.reduction = nn.Sequential(
    nn.Linear(1280, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, self.n_qubits),  # 64 → 4
    nn.Tanh()                       # Bound to [-1, +1]
)
```

This two-stage reduction (1280 → 64 → 4) forces the model to learn which features are most important for retinal diagnosis. The Tanh activation bounds the outputs to the range expected by the quantum angle embedding.

### The Dual-Pathway Design

This is the key architectural innovation of the fundus model. Instead of sending features only through the quantum circuit, we maintain a **parallel classical pathway**:

```python
def forward(self, x):
    # EfficientNet feature extraction
    features = self.backbone.extract_features(x)
    features = self.backbone._avg_pooling(features).flatten(1)
    
    # Reduce to 4 dimensions
    reduced = self.reduction(features)  # (batch, 4)
    
    # Parallel pathways
    quantum_out = self.quantum_layer(reduced)     # (batch, 4)
    classical_out = reduced                        # (batch, 4) — identity
    
    # Concatenate
    combined = torch.cat([classical_out, quantum_out], dim=1)  # (batch, 8)
    
    # Final classification
    output = self.classifier(combined)  # (batch, 1)
    return output
```

**Why dual pathway?**

1. **Redundancy**: If the quantum circuit fails to learn useful representations (e.g., hits a barren plateau), the classical pathway preserves the original signal.
2. **Complementary information**: The quantum pathway captures entangled feature correlations; the classical pathway preserves the raw reduced features. The classifier learns to optimally combine both.
3. **Interpretability**: We can analyse what each pathway contributes by examining gradient flow through the two branches.

### Quantum Circuit: 4-Qubit VQC

```python
n_qubits = 4
n_layers = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    for layer in range(n_layers):
        for qubit in range(n_qubits):
            qml.RY(weights[layer, qubit, 0], wires=qubit)
            qml.RZ(weights[layer, qubit, 1], wires=qubit)
        for qubit in range(n_qubits):
            qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
    
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

**Trainable quantum parameters**: 6 layers × 4 qubits × 2 angles = **48 parameters**

### Classification Head

```python
self.classifier = nn.Sequential(
    nn.Linear(self.n_qubits * 2, 32),  # 8 → 32
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Dropout(0.3),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)                    # Binary logit
)
```

The input dimension is `n_qubits * 2 = 8` because we concatenate the classical (4) and quantum (4) pathway outputs.

---

## Training Procedure

### Dataset

ODIR-5K fundus images, binary split:
- **Class 0**: Normal (N label) — ~2,500 images
- **Class 1**: Abnormal (all other labels) — ~2,500 images

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| **Epochs** | 20 | EfficientNet converges faster with pretrained weights |
| **Batch size** | 16 | Smaller due to larger model memory footprint |
| **Optimiser** | AdamW | Weight decay helps with EfficientNet fine-tuning |
| **Learning rate** | 1 × 10⁻⁴ | Lower than OCT; pretrained features need gentle updates |
| **Weight decay** | 1 × 10⁻⁴ | L2 regularisation via AdamW |
| **Scheduler** | CosineAnnealingLR | Smooth decay prevents learning rate cliff |
| **T_max** | 20 (same as epochs) | Full cosine cycle over training |
| **Gradient clipping** | max_norm = 1.0 | Stabilises quantum gradient flow |
| **Loss function** | BCEWithLogitsLoss | Standard for binary classification |

### Training Strategy

**Phase 1: Frozen Backbone** (optional warmup)
- EfficientNet weights frozen; only train reduction network, quantum circuit, and classifier.
- Purpose: Let the quantum circuit "warm up" before the backbone features start shifting.

**Phase 2: Full Fine-Tuning**
- All parameters unfrozen; entire model trains end-to-end.
- The low learning rate (1e-4) prevents catastrophic forgetting of pretrained features.
- CosineAnnealingLR gradually reduces the learning rate toward zero.

### Training Loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

for epoch in range(20):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
```

### Data Augmentation

Applied during training only (see Document 03):

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## Training Results

### Final Metrics

| Metric | Value |
|---|---|
| **Validation Accuracy** | 68.26% |
| **Validation AUC** | 0.7503 |
| **Training Loss** | 0.4721 |
| **Total Training Time** | ~4 hours (20 epochs) |
| **Model Size** | ~21 MB (dominated by EfficientNet weights) |

### Performance Analysis

The 68.26% accuracy is below the 93% clinical target. Key factors:

1. **Limited dataset**: ODIR-5K provides only ~5,000 images. EfficientNet was designed for million-image datasets; 5K is insufficient for full fine-tuning.

2. **Heterogeneous positive class**: The "abnormal" class includes diabetic retinopathy, glaucoma, cataracts, AMD, and more — very different pathologies with very different visual signatures.

3. **Domain gap**: EfficientNet-B0 pretrained features learned from natural images (dogs, cars, landscapes). Retinal images are a fundamentally different domain.

4. **Quantum circuit expressivity**: 4 qubits with 6 layers may not have enough capacity to represent the complexity needed.

### Paths to Improvement

- **Larger fundus datasets**: EyePACS (80K+ images), APTOS 2019 (3.7K), combined datasets
- **Domain-specific pretraining**: Use RetFound or other retinal foundation models instead of ImageNet EfficientNet
- **More qubits**: 8 or 12 qubits for the fundus pipeline
- **Multi-task learning**: Train classification and segmentation jointly
- **Progressive unfreezing**: Gradually unfreeze EfficientNet layers from top to bottom

---

## OCT vs. Fundus: Architecture Comparison

| Aspect | OCT (8-Qubit) | Fundus (4-Qubit Hybrid) |
|---|---|---|
| **Input** | 64-dim feature vector | 224×224 RGB image |
| **Feature extraction** | Handcrafted (gradient, LBP, etc.) | EfficientNet-B0 (pretrained CNN) |
| **Feature dimension** | 64 → 8 (pre-net) | 1280 → 4 (reduction net) |
| **Qubits** | 8 | 4 |
| **Variational layers** | 8 | 6 |
| **Quantum parameters** | 128 | 48 |
| **Dual pathway** | No (quantum only) | Yes (classical + quantum) |
| **Total parameters** | ~3K | ~5.3M (mostly EfficientNet) |
| **Best accuracy** | 79.66% | 68.26% |
| **Best AUC** | 0.8473 | 0.7503 |
| **Training dataset** | Kermany2018 (84K images) | ODIR-5K (5K images) |

The OCT model outperforms the fundus model primarily due to the 17× larger training dataset — not because of architectural superiority. With equivalent data, the hybrid EfficientNet + quantum approach would likely outperform the handcrafted-feature approach.

---

## Explainability: Grad-CAM

A major advantage of the EfficientNet backbone is that it enables **Grad-CAM** (Gradient-weighted Class Activation Mapping) — a technique that highlights which regions of the input image most influenced the prediction.

For the fundus model, Grad-CAM is computed on the final convolutional layer (`_conv_head`) of EfficientNet-B0:

```python
# Simplified Grad-CAM flow:
# 1. Forward pass through model
# 2. Compute gradients of output w.r.t. _conv_head activations
# 3. Global average pool the gradients
# 4. Weight the activations by the pooled gradients
# 5. ReLU → normalise → resize to input dimensions
# 6. Overlay as jet colormap on original image
```

This produces a heatmap showing which parts of the fundus image the model focused on — ideally the macular region for CSCR detection. Grad-CAM is covered in full detail in Document 10.

---

## Model Save and Load

### Saving

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'val_auc': best_auc
}, 'weights/fundus_quantum.pth')
```

### Loading for Inference

```python
model = QuantumFundusModel(n_qubits=4, n_layers=6)
checkpoint = torch.load('weights/fundus_quantum.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

The saved file is ~21 MB — almost entirely EfficientNet-B0 weights, with the quantum circuit adding negligible size.

The next document (07) covers the U-Net segmentation pipeline, a fully classical architecture for macular region delineation.
