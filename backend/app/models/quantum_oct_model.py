"""
Quantum OCT Classification Model
8-qubit PennyLane circuit with angle embedding and basic entangler layers
for binary classification of OCT images (Normal vs CSR).
"""
import torch
import torch.nn as nn
import pennylane as qml
import numpy as np


NUM_QUBITS = 8
NUM_LAYERS = 8   # Doubled from 4 — deeper circuit for better expressivity
NUM_FEATURES = 64


# Create the quantum device
dev = qml.device("default.qubit", wires=NUM_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    8-qubit quantum circuit.
    - Angle embedding of input features (batched 8 at a time)
    - Basic entangler layers with trainable parameters
    - Measurement: expectation value of PauliZ on each qubit
    """
    # Angle embedding — encode 8 input values into qubit rotations
    qml.AngleEmbedding(inputs, wires=range(NUM_QUBITS))

    # Variational layers
    for layer_idx in range(NUM_LAYERS):
        # Rotation gates with trainable parameters
        for qubit in range(NUM_QUBITS):
            qml.RY(weights[layer_idx, qubit, 0], wires=qubit)
            qml.RZ(weights[layer_idx, qubit, 1], wires=qubit)

        # Entanglement — CNOT ring
        for qubit in range(NUM_QUBITS):
            qml.CNOT(wires=[qubit, (qubit + 1) % NUM_QUBITS])

    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


class QuantumLayer(nn.Module):
    """Wraps the PennyLane quantum circuit as a PyTorch module."""

    def __init__(self):
        super().__init__()
        # Trainable quantum parameters: (NUM_LAYERS, NUM_QUBITS, 2) for RY and RZ
        self.weights = nn.Parameter(
            torch.randn(NUM_LAYERS, NUM_QUBITS, 2) * 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, 8) — 8 features per circuit eval.
        Returns:
            Tensor of shape (batch, 8) — one measurement per qubit.
        """
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            result = quantum_circuit(x[i], self.weights)
            outputs.append(torch.stack(result))
        return torch.stack(outputs).float()  # Cast to float32


class QuantumOCTClassifier(nn.Module):
    """
    Full OCT classification model:
    1. Classical feature reduction: 64 → 8 (to match qubit count)
    2. Quantum circuit: 8 → 8
    3. Classical output head: 8 → 1 (binary classification)
    """

    def __init__(self):
        super().__init__()

        # Pre-quantum: reduce 64 features to 8
        self.pre_net = nn.Sequential(
            nn.Linear(NUM_FEATURES, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, NUM_QUBITS),
            nn.Tanh(),  # Bound inputs to [-1, 1] for angle embedding
        )

        # Quantum layer
        self.quantum = QuantumLayer()

        # Post-quantum: classify from measurements (wider head for deeper circuit)
        self.post_net = nn.Sequential(
            nn.Linear(NUM_QUBITS, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 64) — extracted OCT features.
        Returns:
            Tensor of shape (batch, 1) — logits.
        """
        x = self.pre_net(x)
        x = self.quantum(x)
        x = self.post_net(x)
        return x

    def predict(self, x: torch.Tensor) -> dict:
        """Run inference and return prediction with confidence."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            prob = torch.sigmoid(logits).item()
            prediction = "CSR" if prob >= 0.5 else "Normal"
            confidence = prob if prob >= 0.5 else 1.0 - prob

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probability": float(prob),
        }
