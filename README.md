# RETINA-Q

**Hybrid Quantum-Classical Retinal Disease Diagnosis System**

> **Disclaimer:** This is an educational and research project exploring Quantum Machine Learning (QML) in Medical Imaging. It is **NOT** a medical device, **NOT** FDA-approved, and must **NEVER** be used to diagnose real patients. Always consult a qualified ophthalmologist for medical decisions.

RETINA-Q is a dual-modality retinal diagnostic system that combines quantum computing (PennyLane) with classical deep learning (PyTorch) to classify OCT and Fundus images for Central Serous Retinopathy (CSR/CSCR). It features a clean terminal-aesthetic web UI, asynchronous inference, and full explainability.

---

## Core Capabilities

| Feature | Description |
|---------|-------------|
| **OCT Classification** | Normal vs CSR — 8-qubit, 8-layer quantum circuit via PennyLane |
| **Fundus Classification** | Healthy vs CSCR — EfficientNet-B0 + 4-qubit quantum layer |
| **Macular Segmentation** | U-Net with BCE + Dice + Tversky loss (conditional on Fundus) |
| **Explainability** | Grad-CAM heatmaps (Fundus) + Feature Importance maps (OCT) |
| **Async Processing** | Celery workers with Redis broker, SSE real-time streaming |
| **Physician Feedback** | Accept/Override loop with image quarantine for retraining |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Quantum ML | PennyLane (`default.qubit` simulator) |
| Classical ML | PyTorch + EfficientNet-B0 |
| Segmentation | U-Net (31M params) |
| Explainability | Grad-CAM + gradient-based feature importance |
| Backend API | FastAPI + Pydantic v2 |
| Async Workers | Celery + Redis |
| Frontend | Next.js 14 + Tailwind CSS + Recharts |
| Database | PostgreSQL 16 |
| Model Registry | MLflow v2.10 |
| Infrastructure | Docker + Docker Compose |

---

## Prerequisites

Before running RETINA-Q, ensure you have the following installed:

| Requirement | Minimum Version | Check Command |
|-------------|----------------|---------------|
| **Docker** | 24.0+ | `docker --version` |
| **Docker Compose** | v2.20+ (plugin) | `docker compose version` |
| **Git** | 2.30+ | `git --version` |

> **Note:** On Linux, Docker commands may require `sudo` unless your user is in the `docker` group. All commands below use `sudo` for compatibility. If you've configured rootless Docker, omit `sudo`.

### Adding your user to the Docker group (optional, avoids sudo)

```bash
sudo usermod -aG docker $USER
newgrp docker
# Log out and back in for this to take effect permanently
```

---

## Quick Start (Docker — Recommended)

### 1. Clone the repository

```bash
git clone <repository-url> retina-q
cd retina-q
```

### 2. Verify pretrained weights exist

The model weights must be present in `backend/weights/`. The repository ships with them:

```bash
ls -lh backend/weights/
# Expected files:
#   fundus_quantum.pth    (~16 MB)
#   oct_quantum.pth       (~300 KB)
#   unet_segmentation.pth (~120 MB)
```

If weights are missing, you must either train the models (see [Training](#training-models)) or obtain them from the project maintainer.

### 3. Build and launch

```bash
# One-command launch (builds + starts all 6 services)
sudo docker compose up --build -d
```

This starts:

| Service | Container | Port | Purpose |
|---------|-----------|------|---------|
| Backend API | `retinaq-backend` | 8000 | FastAPI REST endpoints |
| Celery Worker | `retinaq-celery-worker` | — | Async model inference |
| Frontend | `retinaq-frontend` | 3000 | Next.js web dashboard |
| PostgreSQL | `retinaq-postgres` | 5432 | Persistent storage |
| Redis | `retinaq-redis` | 6379 | Message broker + result backend |
| MLflow | `retinaq-mlflow` | 5000 | Model registry (optional) |

### 4. Verify all services are running

```bash
# Check containers
sudo docker compose ps

# Health check
curl http://localhost:8000/health
# Expected: {"status":"ok","service":"RETINA-Q","version":"2.0.0"}

# Check frontend
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000
# Expected: 200
```

### 5. Access the application

| URL | Description |
|-----|-------------|
| http://localhost:3000 | Web dashboard — upload images, run diagnosis |
| http://localhost:8000/docs | Interactive API docs (Swagger UI) |
| http://localhost:5000 | MLflow model registry |

### 6. Test with sample images

The repository includes test images in `test_images/`:

```bash
# Test OCT classification
curl -X POST -F "file=@test_images/oct/NORMAL-450532-1.jpeg" http://localhost:8000/api/predict/oct

# Test Fundus classification
curl -X POST -F "file=@test_images/fundus/2909_right.jpg" http://localhost:8000/api/predict/fundus

# Both return a job_id. Poll for results:
curl http://localhost:8000/api/jobs/<job_id>
```

### 7. Stop the stack

```bash
sudo docker compose down

# To also remove volumes (database data, uploads):
sudo docker compose down -v
```

---

## Alternative: Native Launch (No Docker)

If you prefer running services directly on your machine:

### Backend

```bash
cd backend

# Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start Redis (required for Celery)
# Install Redis separately: sudo apt install redis-server && sudo systemctl start redis

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# In a separate terminal, start the Celery worker
celery -A app.celery_app worker --loglevel=info --concurrency=2
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Development mode
npm run dev

# Or production build
npm run build && npm start
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug logging |
| `HOST` | `0.0.0.0` | API bind host |
| `PORT` | `8000` | API bind port |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `CELERY_BROKER_URL` | `redis://localhost:6379/0` | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/1` | Celery results |
| `MLFLOW_ENABLED` | `false` | Enable MLflow tracking |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend → Backend URL |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict/oct` | OCT quantum classification |
| `POST` | `/api/predict/fundus` | Fundus hybrid classification + optional segmentation |
| `POST` | `/api/segment` | Standalone U-Net macular segmentation |
| `GET` | `/api/jobs/{id}` | Poll async job status |
| `GET` | `/api/jobs/{id}/stream` | SSE real-time job updates |
| `POST` | `/api/feedback` | Physician verdict + correction logging |
| `GET` | `/health` | System health check |

All prediction endpoints accept multipart form upload with a `file` field (JPEG/PNG). Responses include `job_id` for polling. Add `?async_mode=true` for SSE streaming.

---

## Project Structure

```
retina-q/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application entry point
│   │   ├── celery_app.py           # Celery worker configuration
│   │   ├── config.py               # Environment configuration
│   │   ├── tasks.py                # Celery task definitions
│   │   ├── models/
│   │   │   ├── quantum_oct_model.py    # 8-qubit OCT quantum circuit
│   │   │   ├── quantum_fundus_model.py # EfficientNet + 4-qubit hybrid
│   │   │   └── unet_model.py          # U-Net segmentation model
│   │   ├── routes/
│   │   │   ├── predict.py          # /api/predict/* endpoints
│   │   │   ├── segment.py          # /api/segment endpoint
│   │   │   ├── jobs.py             # Job status polling + SSE
│   │   │   └── feedback.py         # Physician feedback endpoint
│   │   ├── services/
│   │   │   ├── inference.py        # Model loading + prediction orchestration
│   │   │   ├── explainability.py   # Grad-CAM + feature importance
│   │   │   └── mlflow_registry.py  # MLflow integration
│   │   └── utils/
│   │       ├── image_processing.py     # Image preprocessing
│   │       └── oct_feature_extractor.py # 64-feature OCT extraction
│   ├── weights/                    # Pretrained model weights (.pth)
│   ├── scripts/                    # Training scripts
│   └── tests/                      # Backend tests
├── frontend/
│   ├── app/
│   │   ├── layout.tsx              # Root layout with terminal header
│   │   ├── page.tsx                # Main diagnosis page
│   │   ├── dashboard/page.tsx      # System dashboard
│   │   └── globals.css             # Terminal-aesthetic styles
│   ├── package.json
│   └── Dockerfile
├── test_images/                    # Sample OCT and Fundus images
│   ├── oct/                        # 3 normal OCT scans
│   └── fundus/                     # 3 fundus photographs
├── docs/                           # Architecture & research docs
├── docker-compose.yml              # Full stack orchestration
├── start_docker_stack.sh           # One-click Docker launch
└── start_demo.sh                   # Native (no Docker) launch
```

---

## Model Architecture

### OCT Pipeline (Quantum)
```
Input Image → 64 Statistical Features (gradient, histogram, LBP, texture, moments)
  → Linear(64→32) → ReLU → BN → Dropout
  → Linear(32→8) → Tanh
  → 8-Qubit Quantum Circuit (8 variational layers, CNOT ring entanglement)
  → Linear(8→32) → ReLU → BN → Dropout
  → Linear(32→16) → Linear(16→1) → Sigmoid
```

### Fundus Pipeline (Hybrid)
```
Input Image (224×224) → EfficientNet-B0 → 1,280 features
  → Linear(1280→64) → ReLU → Dropout
  → Linear(64→4) → Tanh
  → 4-Qubit Quantum Circuit (6 layers, RY/RZ + CNOT ring)
  → Concat(classical_4, quantum_4) = 8-dim
  → Linear(8→32) → ReLU → BN
  → Linear(32→16) → Linear(16→1) → Sigmoid
```

### Segmentation Pipeline (U-Net)
```
Green Channel → CLAHE → U-Net Encoder (1→64→128→256→512)
  → Bottleneck (512→1024)
  → Decoder with skip connections
  → Binarize → Largest connected component → Morphological close/open
```

---

## Training Models

Training scripts are in `backend/scripts/`. They require a GPU and appropriate datasets.

```bash
cd backend

# Train OCT classifier
python scripts/train_oct.py

# Train Fundus classifier
python scripts/train_fundus.py

# Train U-Net segmentation
python scripts/train_segmentation.py
```

After training, copy the `.pth` files to `backend/weights/` and rebuild the containers:

```bash
sudo docker compose up --build -d backend celery-worker
```

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| OCT Accuracy | ≥ 92% | Binary: Normal vs CSR |
| Fundus Accuracy | ≥ 93% | Binary: Healthy vs CSCR |
| Dice Score | ≥ 0.90 | U-Net macular segmentation |
| ROC-AUC | ≥ 0.95 | Both pipelines |
| Inference Time | < 2 sec | Per image (CPU) |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `permission denied` on Docker | Use `sudo` or add user to docker group |
| Frontend shows "Backend offline" | Wait 10s after `docker compose up`, check `curl localhost:8000/health` |
| Job stays in "processing" | Check Celery worker: `sudo docker logs retinaq-celery-worker` |
| Celery not connecting | Ensure Redis is running: `sudo docker logs retinaq-redis` |
| Model weights not found | Verify `backend/weights/` has all 3 `.pth` files |
| Port already in use | `sudo docker compose down` first, or change ports in `docker-compose.yml` |
| Slow inference | Expected on CPU (~10-30s). For faster results, use GPU with `pennylane-lightning[gpu]` |

---

## Documentation

- [Algorithm Workflow Diagram](docs/WORKFLOW_DIAGRAM.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [Quantum Model Specifications](docs/QUANTUM_MODELS.md)
- [Innovation & Quantum Advantage](docs/INNOVATION.md)
- [Project Handoff Guide](docs/HANDOFF.md)
- [Training Infrastructure](docs/training_infrastructure.md)
- [Capacitor Mobile Guide](CAPACITOR_MOBILE_GUIDE.md)

---

## License

This project is for educational and research purposes only.

*Built as an experimental exploration into Quantum AI for medical imaging.*
