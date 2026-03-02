# 04 — Quantum Computing Fundamentals for RETINA-Q

## Introduction

RETINA-Q is a **hybrid quantum-classical** system. But what does that actually mean? This document provides the quantum computing background necessary to understand how the quantum layers in our OCT and fundus models work — no physics PhD required.

We will cover qubits, quantum gates, variational quantum circuits, and the specific PennyLane framework used in this project.

---

## Classical Bits vs. Qubits

### Classical Bits

A classical bit is either 0 or 1. A register of 8 bits can represent exactly one of $2^8 = 256$ possible states at a time.

### Qubits

A qubit can be in a **superposition** of 0 and 1 simultaneously:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

where $\alpha$ and $\beta$ are complex amplitudes satisfying $|\alpha|^2 + |\beta|^2 = 1$.

When we **measure** a qubit, the superposition collapses: we get 0 with probability $|\alpha|^2$ and 1 with probability $|\beta|^2$.

**Why this matters for ML**: A system of $n$ qubits can exist in a superposition of all $2^n$ basis states simultaneously. An 8-qubit system lives in a $2^8 = 256$-dimensional Hilbert space. This means a small number of qubits can represent exponentially complex correlations — which is why quantum circuits can potentially find patterns that classical networks cannot efficiently capture.

---

## Quantum Gates

Just as classical logic gates (AND, OR, NOT) manipulate bits, quantum gates manipulate qubit states through unitary rotations.

### Single-Qubit Rotation Gates

RETINA-Q uses two primary rotation gates:

**RY (Rotation around Y-axis):**

$$RY(\theta) = \begin{pmatrix} \cos(\theta/2) & -\sin(\theta/2) \\ \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}$$

Rotates the qubit state on the Bloch sphere around the Y-axis by angle $\theta$. This controls the amplitude balance between $|0\rangle$ and $|1\rangle$.

**RZ (Rotation around Z-axis):**

$$RZ(\theta) = \begin{pmatrix} e^{-i\theta/2} & 0 \\ 0 & e^{i\theta/2} \end{pmatrix}$$

Applies a relative phase rotation. While it doesn't change measurement probabilities alone, it creates interference effects when combined with other gates.

### Two-Qubit Entanglement Gate

**CNOT (Controlled-NOT):**

$$CNOT = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

CNOT flips the target qubit if and only if the control qubit is $|1\rangle$. This creates **entanglement** — a quantum correlation where the state of one qubit becomes dependent on another. Entanglement is the key resource that separates quantum computing from classical computing.

---

## Variational Quantum Circuits (VQC)

A Variational Quantum Circuit is the quantum analogue of a neural network layer. It has:

1. **Data Encoding**: Classical input data is encoded into qubit states.
2. **Parameterised Gates**: Rotation gates with trainable angles (analogous to neural network weights).
3. **Entanglement**: CNOT gates link qubits together.
4. **Measurement**: The quantum state is measured to produce classical output values.

### The VQC Used in RETINA-Q

Both the OCT and fundus models follow this pattern:

```
Step 1: Angle Embedding
   Input values → RY rotations on each qubit
   (Each qubit is initialised to encode one feature value)

Step 2: Variational Layers (repeated L times)
   For each qubit i:
     RY(θ_i) → RZ(φ_i)     ← trainable parameters
   For each pair (i, i+1):
     CNOT(i, i+1)           ← ring entanglement (last qubit connects to first)

Step 3: Measurement
   Measure ⟨PauliZ⟩ on each qubit → output values in [-1, +1]
```

### Visualising the 8-Qubit OCT Circuit

```
q0: ─RY(x₀)─┤─RY(θ₀)──RZ(φ₀)──●───────────────────────X─┤ ⟨Z⟩
q1: ─RY(x₁)─┤─RY(θ₁)──RZ(φ₁)──X──●────────────────────┤ ⟨Z⟩
q2: ─RY(x₂)─┤─RY(θ₂)──RZ(φ₂)─────X──●─────────────────┤ ⟨Z⟩
q3: ─RY(x₃)─┤─RY(θ₃)──RZ(φ₃)────────X──●──────────────┤ ⟨Z⟩
q4: ─RY(x₄)─┤─RY(θ₄)──RZ(φ₄)───────────X──●───────────┤ ⟨Z⟩
q5: ─RY(x₅)─┤─RY(θ₅)──RZ(φ₅)──────────────X──●────────┤ ⟨Z⟩
q6: ─RY(x₆)─┤─RY(θ₆)──RZ(φ₆)─────────────────X──●─────┤ ⟨Z⟩
q7: ─RY(x₇)─┤─RY(θ₇)──RZ(φ₇)────────────────────X──●──┤ ⟨Z⟩
              └─── repeated for L = 8 layers ───┘
```

The `●──X` symbols represent CNOT gates (control → target). The ring topology means qubit 7 also connects back to qubit 0, creating a circular entanglement pattern.

---

## PennyLane: The Quantum ML Framework

RETINA-Q uses **PennyLane** (by Xanadu) to implement quantum circuits. PennyLane integrates seamlessly with PyTorch, allowing quantum circuits to participate in standard backpropagation training.

### Key Concepts

#### QNode

A `QNode` is PennyLane's quantum function decorator. It wraps a quantum circuit so that it behaves like a differentiable PyTorch function:

```python
import pennylane as qml

dev = qml.device("default.qubit", wires=8)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(8))
    for layer_weights in weights:
        for i in range(8):
            qml.RY(layer_weights[i, 0], wires=i)
            qml.RZ(layer_weights[i, 1], wires=i)
        for i in range(8):
            qml.CNOT(wires=[i, (i + 1) % 8])
    return [qml.expval(qml.PauliZ(i)) for i in range(8)]
```

#### Devices

PennyLane supports multiple quantum backends (simulators and real hardware). RETINA-Q uses:

- **`default.qubit`**: A state-vector simulator that runs on CPU/GPU. It simulates the full quantum state mathematically — no actual quantum hardware needed. This enables exact gradient computation via backpropagation.

#### Differentiation Methods

This was a critical engineering decision:

| Method | How it works | Speed |
|---|---|---|
| `parameter-shift` | Evaluates the circuit twice per parameter to estimate gradients analytically. Exact but requires $2P$ circuit evaluations per gradient step (where $P$ = number of parameters). | Very slow |
| `backprop` | Uses the state-vector simulator's internal representation to compute gradients via standard PyTorch autograd. Only works with simulators, not real hardware. | **~100x faster** |

We switched from `parameter-shift` to `backprop` during development — reducing epoch time from **~100 minutes to ~7 minutes**. This is documented in detail in [docs/model_training.md](../model_training.md).

---

## Why Quantum for Medical Imaging?

### The Theoretical Argument

Quantum circuits naturally operate in exponentially large Hilbert spaces. For $n$ qubits, the circuit operates on a $2^n$-dimensional state vector. This means:

- **8 qubits** → 256-dimensional feature space
- **4 qubits** → 16-dimensional feature space

In these spaces, entanglement creates correlations between features that would require exponentially many parameters to replicate classically. The hypothesis is that certain patterns in medical images (subtle, multi-scale, nonlinear) may be more naturally captured by quantum correlations than by classical matrix multiplications.

### The Practical Advantage

Even on simulators (where there's no computational speedup from quantum mechanics), the VQC architecture provides:

1. **Extreme compression**: 1280 EfficientNet features compressed to 4 qubit measurements — a massive information bottleneck that forces the model to learn the most discriminative features.

2. **Entanglement as feature interaction**: CNOT-based ring entanglement means every feature interacts with every other feature through the circuit depth — providing full feature cross-correlation with very few parameters.

3. **Parameter efficiency**: The 8-qubit OCT circuit with 8 layers has only $8 \times 8 \times 2 = 128$ trainable quantum parameters. Combined with the classical pre/post networks, the total model is much smaller than a comparable classical network.

4. **Research value**: As quantum hardware matures (more qubits, lower noise), models trained on simulators can be directly transferred to real quantum processors for potential speedups.

### The Honest Assessment

Current quantum simulators provide no computational advantage over classical methods. The value of the quantum approach in RETINA-Q is:

- Demonstrating that hybrid quantum-classical pipelines can achieve competitive medical imaging performance
- Establishing quantum-ready architectures for future hardware
- Exploring whether quantum feature entanglement captures clinically relevant patterns differently from classical networks
- Contributing to the nascent field of quantum machine learning for healthcare

---

## Measurement and Output Interpretation

When we measure a qubit in the computational basis, we get the **expectation value** of the Pauli-Z operator:

$$\langle Z \rangle = P(|0\rangle) - P(|1\rangle)$$

This ranges from $-1$ (qubit is definitely $|1\rangle$) to $+1$ (qubit is definitely $|0\rangle$). For an 8-qubit circuit, we get 8 real numbers in $[-1, +1]$ — these become the input to the classical post-processing network.

### Why PauliZ Measurements?

PauliZ is the standard computational basis measurement. Each qubit contributes one real-valued output, interpreted as how "confident" that qubit dimension is about the 0 vs. 1 state. These 8 values collectively encode the quantum circuit's "opinion" about the input features.

---

## Connecting to the Models

| Aspect | OCT Model | Fundus Model |
|---|---|---|
| **Qubits** | 8 | 4 |
| **Variational layers** | 8 | 6 |
| **Quantum parameters** | 128 | 48 |
| **Input encoding** | AngleEmbedding (8 features → 8 qubits) | AngleEmbedding (4 features → 4 qubits) |
| **Entanglement** | Ring CNOT (8 qubits) | Ring CNOT (4 qubits) |
| **Measurement** | PauliZ on all 8 qubits | PauliZ on all 4 qubits |
| **Output dimension** | 8 values in [-1, +1] | 4 values in [-1, +1] |
| **Diff method** | backprop | backprop |
| **Device** | default.qubit | default.qubit |

The next documents (05 and 06) dive deep into the OCT and Fundus model architectures — showing exactly how the classical and quantum components are wired together.
