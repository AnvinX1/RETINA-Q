# RETINA-Q: Project Handoff & Continuity Guide

This document provides all necessary information for a developer or researcher to take over the RETINA-Q project and continue its development.

---

## 1. Project At-A-Glance
RETINA-Q is a dual-modality retinal diagnostic system utilizing Quantum-Classical Hybrid Neural Networks. It triages OCT and Fundus images for Central Serous Retinopathy (CSR/CSCR).

### Current State: **STABLE / V2.0.0 (Quantum Upgrade)**
- **OCT Pipeline**: Fully functional (8-qubit, 8-layer quantum circuit).
- **Fundus Pipeline**: Fully functional (4-qubit, 6-layer quantum circuit + EfficientNet-B0 backbone).
- **Infrastructure**: Fully containerized with Docker (FastAPI + Celery + Redis + Postgres + MLflow).

---

## 2. Technical Stack
- **Frontend**: Next.js (TypeScript) + Tailwind CSS + Lucide Icons.
- **Backend API**: FastAPI (Python 3.11).
- **Inference Engine**: Celery Workers (asynchronous processing).
- **Quantum Library**: PennyLane (using `default.qubit` simulator).
- **Classical ML**: PyTorch + EfficientNet-PyTorch.
- **Message Broker**: Redis.
- **Database**: PostgreSQL.
- **Model Management**: MLflow.

---

## 3. Deployment Guide

### Running Locally (Docker)
The entire stack is managed via Docker Compose.
```bash
cd retina-q
./start_docker_stack.sh
```
- **UI**: [http://localhost:3000](http://localhost:3000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **MLflow**: [http://localhost:5000](http://localhost:5000)

### Updating Models
If you perform a new training run on the remote GPU, follow these steps to deploy:
1.  **SCP weights**: Copy `.pth` files from the remote server to `backend/weights/`.
2.  **Verify Architecture**: Ensure `NUM_LAYERS` and `NUM_QUBITS` in `backend/app/models/` match the training script.
3.  **Rebuild Containers**:
    ```bash
    sudo docker compose up --build -d backend celery-worker
    ```

---

## 4. Key Components & Files

### Quantum Models (`backend/app/models/`)
- `quantum_oct_model.py`: Defines the 8-qubit OCT circuit.
- `quantum_fundus_model.py`: Defines the hybrid EfficientNet + 4-qubit Fundus model.

### Inference Logic (`backend/app/services/`)
- `inference.py`: Orchestrates image loading, preprocessing, and model execution.
- `explainability.py`: Generates Grad-CAM (Fundus) and Feature Importance (OCT) heatmaps.

### Training Scripts (`backend/scripts/`)
- `train_fundus.py`: Full training loop for the hybrid Fundus model.
- `train_oct.py`: Training loop for the OCT feature classifier.

---

## 5. Continuity Checklist (Future Work)

1.  **[ ] GPU Acceleration in Docker**: Currently, Docker inference runs on CPU (using PennyLane simulators). Transition to `pennylane-lightning[gpu]` in the `Dockerfile` for faster processing if a local GPU is available.
2.  **[ ] Model Distillation**: The EfficientNet backbone is quite large. Consider distilling the classical knowledge into a smaller MobileNetV3 to reduce latency further.
3.  **[ ] Multi-Class Support**: Expand the current Healthy vs. CSR/CSCR binary classification to include Diabetic Retinopathy or Glaucoma.
4.  **[ ] Real-QPU Integration**: Swap the PennyLane `default.qubit` device for an Amazon Braket or IBM Quantum hardware provider in `backend/app/models/`.

---

## 6. Contacts & Resources
- **Innovation Whitepaper**: [docs/INNOVATION.md](docs/INNOVATION.md)
- **Quantum Specs**: [docs/QUANTUM_MODELS.md](docs/QUANTUM_MODELS.md)
- **Architecture Diagram**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
