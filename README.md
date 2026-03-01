# 👁️ RETINA-Q

**Hybrid Quantum-Classical Multi-Modal Retinal Disease Diagnosis System**

> ⚠️ **Disclaimer: This is a FUN PROJECT!** ⚠️
>
> *This repository is purely for educational, exploratory, and entertainment purposes. It explores the intersection of Quantum Machine Learning (QML) and Medical Imaging. This system is **NOT** a medical device, is **NOT** FDA-approved, and should **NEVER** be used to diagnose, treat, or make medical decisions for real patients. Always consult a qualified ophthalmologist or healthcare professional for medical advice.*

RETINA-Q is an experimental AI-powered clinical decision support system that integrates quantum machine learning with deep learning for automated retinal diagnosis using OCT and Fundus images.

---

## 🚀 Core Capabilities

- **OCT Binary Classification** (Normal vs CSR) — 8-Qubit Quantum Circuit via PennyLane
- **Fundus Binary Classification** (Healthy vs CSCR) — EfficientNet-B0 + 4-Qubit Quantum Layer
- **Conditional Macular Segmentation** — U-Net with BCE + Dice + Tversky Loss
- **Explainability** — Grad-CAM & Feature Importance Mapping
- **REST API Deployment** — Python FastAPI
- **Modern Dashboard** — Next.js with a black-and-white minimalist Shadcn/UI aesthetic
- **Mobile Ready** — Fully compatible with Ionic Capacitor for native iOS/Android builds

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| ML Framework | PyTorch + PennyLane |
| CNN Backbone | EfficientNet-B0 |
| Segmentation | U-Net |
| Explainability | Grad-CAM + Feature Importance |
| Backend API | FastAPI + Pydantic |
| Frontend | Next.js + Tailwind CSS |
| Database | PostgreSQL |
| Infrastructure | Docker + Docker Compose |

---

## 📚 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Quantum Model Specifications](docs/QUANTUM_MODELS.md)
- [Innovation & Quantum Advantage](docs/INNOVATION.md)
- [Project Handoff Guide](docs/HANDOFF.md)
- [Training Infrastructure](docs/training_infrastructure.md)

---

## 📂 Folder Structure

```
eye/
├── backend/                  # FastAPI & PyTorch models
│   ├── app/models/           # Quantum circuits, ResNet, U-Net classes
│   ├── app/routes/           # API Endpoints
│   ├── weights/              # Pretrained PyTorch weights (.pth)
│   └── requirements.txt      
├── frontend/                 # Next.js 14 Dashboard
│   ├── app/                  # Main UI routes & layout
│   └── package.json          
├── docker-compose.yml        # Full-stack containerization
├── start_demo.sh             # Native quick-start script
├── start_docker_stack.sh     # Docker quick-start script
└── CAPACITOR_MOBILE_GUIDE.md # Guide for mobile app conversion
```

---

## 🚦 Quick Start

There are multiple ways to launch RETINA-Q:

### Option 1: Docker Automation (Recommended)
This will boot the database, the Python/PyTorch AI backend, and the Next.js frontend automatically.
```bash
chmod +x start_docker_stack.sh
./start_docker_stack.sh
# Frontend UI: http://localhost:3000
# Backend API: http://localhost:8000/docs
```

### Option 2: Native Demo Script
If you do not want to use Docker, use the native launch script. Make sure you have `npm` and a python virtual environment activated.
```bash
chmod +x start_demo.sh
./start_demo.sh
```

### Option 3: Remote GPU Training Tracker
If you are running the intensive quantum models on a remote GPU cluster, use the tracking script to monitor the training processes in real-time.
```bash
chmod +x track_training.sh
./track_training.sh
```

---

## 🎯 Performance Targets

| Metric | Target |
|--------|--------|
| OCT Accuracy | ≥ 92% |
| Fundus Accuracy | ≥ 93% |
| Dice Score | ≥ 0.90 |
| ROC-AUC | ≥ 0.95 |
| Inference | < 2 sec |

---

*Built with ❤️ as an experimental exploration into Quantum AI.*
