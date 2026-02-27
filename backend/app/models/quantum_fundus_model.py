"""
Quantum Fundus Classification Model
EfficientNet-B0 backbone + 4-qubit quantum layer for binary classification
of fundus images (Healthy vs CSCR).
"""
import torch
import torch.nn as nn
import pennylane as qml
from efficientnet_pytorch import EfficientNet


NUM_QUBITS = 4
NUM_LAYERS = 3
CLASSICAL_FEATURES = 1280  # EfficientNet-B0 output features
QUANTUM_INPUT_DIM = NUM_QUBITS


# Quantum device
dev = qml.device("default.qubit", wires=NUM_QUBITS)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def fundus_quantum_circuit(inputs, weights):
    """
    4-qubit quantum circuit with RY/RZ rotations and CNOT entanglement.
    """
    # Encode inputs as rotations
    for i in range(NUM_QUBITS):
        qml.RY(inputs[i], wires=i)

    # Variational layers
    for layer_idx in range(NUM_LAYERS):
        for qubit in range(NUM_QUBITS):
            qml.RY(weights[layer_idx, qubit, 0], wires=qubit)
            qml.RZ(weights[layer_idx, qubit, 1], wires=qubit)

        # CNOT entanglement
        for qubit in range(NUM_QUBITS - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
        qml.CNOT(wires=[NUM_QUBITS - 1, 0])  # Close the ring

    return [qml.expval(qml.PauliZ(i)) for i in range(NUM_QUBITS)]


class FundusQuantumLayer(nn.Module):
    """PyTorch wrapper for the 4-qubit fundus quantum circuit."""

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(NUM_LAYERS, NUM_QUBITS, 2) * 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            result = fundus_quantum_circuit(x[i], self.weights)
            outputs.append(torch.stack(result))
        return torch.stack(outputs).float()  # Cast to float32 to match classical branch


class QuantumFundusClassifier(nn.Module):
    """
    Hybrid Quantum-Classical Fundus Classifier:
    1. EfficientNet-B0 backbone → 1280-dim features
    2. Adaptive pooling + reduction → 4-dim (for qubits)
    3. 4-Qubit quantum layer → 4-dim quantum features
    4. Concatenate classical (reduced) + quantum features → 8-dim
    5. Classical head → binary output
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # EfficientNet-B0 backbone (remove classifier head)
        if pretrained:
            self.backbone = EfficientNet.from_pretrained("efficientnet-b0")
        else:
            self.backbone = EfficientNet.from_name("efficientnet-b0")

        # Adaptive pooling (already in EfficientNet but we need features)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Reduce classical features to match quantum input
        self.reducer = nn.Sequential(
            nn.Linear(CLASSICAL_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, QUANTUM_INPUT_DIM),
            nn.Tanh(),
        )

        # Quantum layer
        self.quantum = FundusQuantumLayer()

        # Classifier: concatenated classical (4) + quantum (4) = 8
        self.classifier = nn.Sequential(
            nn.Linear(QUANTUM_INPUT_DIM + NUM_QUBITS, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from EfficientNet backbone."""
        features = self.backbone.extract_features(x)
        features = self.pool(features)
        features = features.flatten(start_dim=1)  # (batch, 1280)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 3, 224, 224) — preprocessed fundus image.
        Returns:
            Tensor of shape (batch, 1) — logits.
        """
        # Classical backbone
        features = self.extract_features(x)  # (batch, 1280)

        # Reduce to quantum input dimension
        reduced = self.reducer(features)  # (batch, 4)

        # Quantum processing
        quantum_out = self.quantum(reduced)  # (batch, 4)

        # Concatenate classical + quantum features
        combined = torch.cat([reduced, quantum_out], dim=1)  # (batch, 8)

        # Classify
        logits = self.classifier(combined)
        return logits

    def predict(self, x: torch.Tensor) -> dict:
        """Run inference and return prediction with confidence."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            prob = torch.sigmoid(logits).item()
            prediction = "CSCR" if prob >= 0.5 else "Healthy"
            confidence = prob if prob >= 0.5 else 1.0 - prob

        return {
            "prediction": prediction,
            "confidence": float(confidence),
            "probability": float(prob),
        }
