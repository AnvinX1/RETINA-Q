# 05 — OCT Quantum Model: Architecture & Training

## Introduction

The OCT classification pipeline is RETINA-Q's primary quantum model — a full Variational Quantum Circuit (VQC) with 8 qubits and 8 variational layers. It takes a 224×224 greyscale OCT scan, extracts 64 handcrafted features, compresses them through a classical pre-network, processes them through the quantum circuit, and passes the quantum measurements through a classical post-network to produce a binary diagnosis.

This document covers the complete architecture, the training procedure, the results achieved, and the key engineering decisions made along the way.

---

## End-to-End Pipeline

```
OCT Image (224×224, greyscale)
        │
        ▼
┌─────────────────────────────────┐
│     Feature Extraction          │
│  64-dimensional vector:         │
│  • 16 gradient features         │
│  • 16 histogram features        │
│  • 16 LBP texture features      │
│  • 16 statistical moments       │
└───────────────┬─────────────────┘
                │ (64,)
                ▼
┌─────────────────────────────────┐
│     Pre-Network (Classical)     │
│  Linear(64 → 32) → ReLU → BN   │
│  → Dropout(0.3)                 │
│  Linear(32 → 8) → Tanh          │
└───────────────┬─────────────────┘
                │ (8,)  ← values in [-1, +1]
                ▼
┌─────────────────────────────────┐
│   8-Qubit Quantum Circuit       │
│  AngleEmbedding → 8 layers of:  │
│    RY/RZ rotations + ring CNOT  │
│  → PauliZ measurement (8 qubits)│
└───────────────┬─────────────────┘
                │ (8,)  ← values in [-1, +1]
                ▼
┌─────────────────────────────────┐
│    Post-Network (Classical)     │
│  Linear(8 → 32) → ReLU → BN    │
│  → Dropout(0.3)                 │
│  Linear(32 → 16) → ReLU         │
│  Linear(16 → 1) → Sigmoid       │
└───────────────┬─────────────────┘
                │ (1,)  ← probability in [0, 1]
                ▼
         Normal (< 0.5) or CSR (≥ 0.5)
```

---

## Architecture Deep Dive

### Pre-Network: Feature Compression

The 64-dimensional feature vector must be compressed to 8 values — one per qubit. The pre-network performs this learned dimensionality reduction:

```python
self.pre_net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Dropout(0.3),
    nn.Linear(32, self.n_qubits),  # 32 → 8
    nn.Tanh()                       # Bound to [-1, +1]
)
```

**Why Tanh?** The quantum circuit uses AngleEmbedding, which interprets inputs as rotation angles. Tanh bounds the values to [-1, +1], which maps to a reasonable range of qubit rotations (roughly ±57° on the Bloch sphere). Unbounded inputs would cause the circuit to "wrap around" multiple times, creating a many-to-one mapping that hinders learning.

### Quantum Circuit: 8-Qubit VQC

The circuit definition in PennyLane:

```python
n_qubits = 8
n_layers = 8
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    # Encode 8 classical values as qubit rotation angles
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    
    # 8 variational layers
    for layer in range(n_layers):
        # Parameterised single-qubit rotations
        for qubit in range(n_qubits):
            qml.RY(weights[layer, qubit, 0], wires=qubit)
            qml.RZ(weights[layer, qubit, 1], wires=qubit)
        
        # Ring entanglement: CNOT chain with wrap-around
        for qubit in range(n_qubits):
            qml.CNOT(wires=[qubit, (qubit + 1) % n_qubits])
    
    # Measure all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

**Trainable parameters**: 8 layers × 8 qubits × 2 angles (RY, RZ) = **128 quantum parameters**

**Ring entanglement**: Each qubit is CNOTed with its neighbour, and qubit 7 connects back to qubit 0. After 8 layers, every qubit has been entangled with every other qubit multiple times, enabling complex feature correlations.

### Post-Network: Classification Head

The 8 quantum measurement values feed into the classification head:

```python
self.post_net = nn.Sequential(
    nn.Linear(self.n_qubits, 32),  # 8 → 32
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Dropout(0.3),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)               # Binary logit
)
```

The final layer outputs a single logit. During training, `BCEWithLogitsLoss` applies sigmoid internally. During inference, `torch.sigmoid()` converts to a probability.

---

## Training Procedure

### Dataset Setup

```python
# Load from Kermany2018 dataset
# Binary labelling: NORMAL → 0, all others (CNV/DME/DRUSEN) → 1
# Feature extraction: 64-dim vector per image (computed on-the-fly)
```

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| **Epochs** | 30 | Sufficient for convergence with early stopping |
| **Batch size** | 32 | Balances GPU memory with gradient stability |
| **Optimiser** | Adam | Adaptive learning rates suit mixed classical-quantum training |
| **Learning rate** | 5 × 10⁻⁴ | Found via grid search; lower rates stall, higher rates diverge |
| **Weight decay** | 0 | Dropout provides sufficient regularisation |
| **Gradient clipping** | max_norm = 1.0 | Prevents exploding gradients from quantum circuit |
| **Dropout** | 0.3 | Applied in both pre and post networks |
| **Loss function** | BCEWithLogitsLoss | Numerically stable binary cross-entropy |

### Training Loop

```python
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

for epoch in range(30):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_preds, val_labels = [], []
        for features, labels in val_loader:
            outputs = torch.sigmoid(model(features).squeeze())
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    auc = roc_auc_score(val_labels, val_preds)
    acc = accuracy_score(val_labels, [1 if p > 0.5 else 0 for p in val_preds])
    print(f"Epoch {epoch}: Loss={loss:.4f}, AUC={auc:.4f}, Acc={acc:.4f}")
```

### The Gradient Breakthrough

The single most impactful engineering decision was switching the quantum differentiation method:

| Method | Epoch Time | Gradient Quality |
|---|---|---|
| `parameter-shift` | ~100 minutes | Exact (analytical) |
| `backprop` | ~7 minutes | Exact (autograd) |

Both methods produce exact gradients. The `parameter-shift` rule evaluates the circuit $2P$ times per gradient step (where $P$ is parameters). With 128 quantum parameters, that's 256 circuit evaluations per step — incredibly slow.

The `backprop` method leverages the fact that `default.qubit` is a state-vector simulator running in PyTorch. It can backpropagate through the simulated quantum operations using standard autograd — same gradients, 100× faster.

**Trade-off**: `backprop` only works on simulators. On real quantum hardware, `parameter-shift` would be necessary. But for training and development, `backprop` is the only practical choice.

---

## Training Results

### Convergence Behaviour

```
Epoch  1: Loss=0.6932, AUC=0.5012, Acc=50.23%  ← random
Epoch  5: Loss=0.5841, AUC=0.6234, Acc=61.45%  ← learning starts
Epoch 10: Loss=0.4523, AUC=0.7156, Acc=68.92%  ← steady improvement
Epoch 15: Loss=0.3876, AUC=0.7745, Acc=73.18%  ← approaching plateau
Epoch 20: Loss=0.3412, AUC=0.8102, Acc=76.54%  ← diminishing returns
Epoch 25: Loss=0.3198, AUC=0.8356, Acc=78.41%  ← near convergence
Epoch 30: Loss=0.3054, AUC=0.8473, Acc=79.66%  ← final
```

### Final Metrics

| Metric | Value |
|---|---|
| **Validation Accuracy** | 79.66% |
| **Validation AUC** | 0.8473 |
| **Training Loss** | 0.3054 |
| **Total Training Time** | ~3.5 hours (30 epochs × 7 min/epoch) |
| **Model Size** | ~47 KB (saved weights) |

### Performance Analysis

The 79.66% accuracy and 0.8473 AUC represent a solid baseline for an 8-qubit quantum model, but fall short of the ≥92% clinical target. Contributing factors:

1. **Feature extraction bottleneck**: The 64-dimensional handcrafted features, while domain-informed, may not capture all discriminative information in the OCT scans.

2. **Binary grouping**: Combining CNV, DME, and DRUSEN into a single "abnormal" class creates a heterogeneous positive class.

3. **Simulation overhead**: The `default.qubit` simulator provides perfect, noiseless quantum operations. Real quantum hardware introduces noise that could actually help with generalisation (analogous to training with noise as regularisation).

4. **Circuit depth**: 8 layers with 8 qubits is relatively shallow. Deeper circuits could capture more complex patterns but risk barren plateaus.

### Paths to Improvement

- **Larger training set**: Use the full 84K Kermany dataset with proper class balancing
- **Deeper circuits**: 12–16 variational layers (with careful initialisation to avoid barren plateaus)
- **Data-reuploading**: Re-encode the input features at each layer, not just the first
- **Hybrid features**: Combine handcrafted features with CNN-extracted features
- **Class-specific training**: Train separate binary classifiers for NORMAL vs. each pathology type

---

## Model Save and Load

### Saving

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'val_auc': best_auc
}, 'weights/oct_quantum.pth')
```

### Loading for Inference

```python
model = QuantumOCTModel(n_qubits=8, n_layers=8)
checkpoint = torch.load('weights/oct_quantum.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

The saved weights file at `weights/oct_quantum.pth` is approximately 47 KB — remarkably small for a medical imaging classifier. This is a direct benefit of the quantum architecture: 128 quantum parameters + ~3K classical parameters = very compact model.

---

## Inference Flow (Production)

When a user uploads an OCT image through the API:

1. **Image saved** to `uploads/{job_id}.jpg`
2. **Feature extraction**: 64-dim vector computed via `oct_feature_extractor.py`
3. **Model forward pass**: Pre-net → Quantum circuit → Post-net → Sigmoid probability
4. **Explainability**: Gradient-based feature importance computed (see Document 10)
5. **Result returned**: `{prediction: "Normal"/"CSR", confidence: 0.87, heatmap: "base64..."}`

The model runs on CPU for inference (no GPU required), with typical latency of 10–30 seconds — dominated by the quantum circuit simulation.

The next document (06) covers the Fundus Hybrid Model, which takes a fundamentally different architectural approach.
