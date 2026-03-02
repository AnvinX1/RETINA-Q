# 01 — Project Overview: RETINA-Q

## What Is RETINA-Q?

RETINA-Q (Retinal Imaging Through Intelligent Neural Analysis — Quantum) is a hybrid quantum-classical diagnostic system built to detect **Central Serous Chorioretinopathy (CSCR)** from retinal scans. CSCR is a condition where fluid accumulates beneath the retina, causing visual distortion and, if left untreated, permanent vision loss. Early and accurate detection is critical — and that is exactly what RETINA-Q was designed to do.

The system accepts two types of retinal images:

- **OCT (Optical Coherence Tomography)** — cross-sectional scans showing the layered structure of the retina.
- **Fundus photographs** — colour images of the back of the eye captured through a fundoscope.

For each scan, RETINA-Q returns a binary classification (Healthy vs. CSCR), a confidence score, and a visual explainability overlay (heatmap) showing what regions of the image the model focused on. For fundus images, an additional macular segmentation mask is generated to delineate the area of interest.

---

## Why Does This Project Exist?

Retinal disease screening in many healthcare settings still depends on a specialist manually reviewing scans — a process that is time-consuming, subject to fatigue-related errors, and unavailable in underserved communities. Automated AI screening can:

1. **Reduce diagnostic latency** — results in seconds rather than days.
2. **Scale beyond expert availability** — one model can serve thousands of clinics.
3. **Provide objective, reproducible assessments** — eliminating inter-observer variability.
4. **Offer visual reasoning** — heatmaps let clinicians understand *why* the model flagged a scan.

RETINA-Q goes a step further by incorporating **quantum computing** into the machine learning pipeline. The quantum layers serve as highly expressive feature-entanglement bottlenecks that can capture complex correlations in compressed feature spaces — a research frontier where quantum advantage for medical imaging is being actively explored.

---

## The Clinical Problem: Central Serous Chorioretinopathy

CSCR is characterised by serous detachment of the neurosensory retina. Key clinical facts:

- **Prevalence**: Affects roughly 1 in 10,000 people, predominantly males aged 20–50.
- **Symptoms**: Blurred central vision, metamorphopsia (distorted straight lines), micropsia (objects appear smaller).
- **OCT findings**: Sub-retinal fluid pockets, retinal pigment epithelium (RPE) detachment, increased retinal thickness.
- **Fundus findings**: Focal leakage points, serous elevation of the macula, altered pigmentation.
- **Risk**: Chronic CSCR leads to photoreceptor atrophy and irreversible central vision loss.

Early detection from routine scans — before the patient notices symptoms — is the highest-value clinical intervention. RETINA-Q targets exactly this screening use case.

---

## System Architecture at a Glance

The project is composed of five interconnected layers:

```
┌──────────────────────────────────────────────────────────┐
│                      FRONTEND                            │
│  Next.js 14 · React 18 · Tailwind CSS · Terminal UI      │
│  Image Upload → Real-Time Progress (SSE) → Results View  │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼───────────────────────────────────┐
│                    BACKEND API                           │
│  FastAPI · Pydantic v2 · Uvicorn · CORS                  │
│  Routes: predict / segment / feedback / patients / jobs  │
└──────────┬───────────────────────────┬───────────────────┘
           │ Celery Task Dispatch      │ SQLAlchemy ORM
┌──────────▼──────────┐    ┌───────────▼──────────────────┐
│    ASYNC WORKERS    │    │        DATABASE              │
│  Celery + Redis     │    │  PostgreSQL 16 · Patients    │
│  OCT / Fundus /     │    │  Scans · Feedback Logs       │
│  Segmentation tasks │    └──────────────────────────────┘
└──────────┬──────────┘
           │ PyTorch + PennyLane
┌──────────▼──────────────────────────────────────────────┐
│                   ML MODELS                             │
│  OCT:   8-Qubit Variational Quantum Classifier          │
│  Fundus: EfficientNet-B0 + 4-Qubit Hybrid Classifier   │
│  Segmentation: Classical U-Net (Dice = 0.9663)          │
│  Explainability: Grad-CAM · Feature Importance Mapping  │
└─────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Quantum ML** | PennyLane 0.35+, `default.qubit` simulator, `diff_method="backprop"` |
| **Classical ML** | PyTorch 2.0+, EfficientNet-B0, TorchVision |
| **Segmentation** | Custom U-Net, BCE + Dice + Tversky loss |
| **Explainability** | Grad-CAM (torchcam), gradient-based feature importance |
| **Backend** | FastAPI, Uvicorn, Pydantic v2 |
| **Async Processing** | Celery 5.3+, Redis 7 |
| **Database** | PostgreSQL 16, SQLAlchemy |
| **Frontend** | Next.js 14 App Router, React 18, Tailwind CSS, Recharts |
| **Desktop** | Electron 33, Electron Builder |
| **Containerisation** | Docker, Docker Compose (6 services) |
| **MLOps** | MLflow 2.10 model registry |
| **Logging** | Loguru with file rotation |

---

## What Makes RETINA-Q Different?

1. **Quantum-classical hybrid models** — not just classical deep learning, but true quantum circuit integration via PennyLane, exploring the frontier of quantum machine learning for medical diagnostics.

2. **Multi-modal input** — handles both OCT and fundus images through separate, specialised pipelines.

3. **End-to-end explainability** — every prediction comes with a visual heatmap overlay and feature importance breakdown, designed for clinician trust.

4. **Asynchronous architecture** — all inference runs on Celery workers with real-time Server-Sent Events streaming, so the API never blocks.

5. **Doctor-in-the-loop feedback** — clinicians can confirm or override predictions, with rejected samples automatically quarantined for future retraining.

6. **Cross-platform deployment** — Docker stack for servers, Electron for desktop, and Capacitor.js for mobile (iOS/Android).

---

## Performance Summary

| Model | Metric | Current | Target |
|---|---|---|---|
| OCT (8-Qubit) | Accuracy | 79.66% | ≥ 92% |
| OCT (8-Qubit) | AUC | 0.8473 | ≥ 0.95 |
| Fundus (4-Qubit Hybrid) | Accuracy | 68.26% | ≥ 93% |
| Fundus (4-Qubit Hybrid) | AUC | 0.7503 | ≥ 0.95 |
| U-Net Segmentation | Dice Score | 0.9663 | ≥ 0.90 |

These numbers reflect training on publicly available datasets (Kermany2018 for OCT, ODIR-5K for Fundus). Performance is expected to improve significantly with larger, curated clinical datasets and deeper quantum circuits on real quantum hardware.

---

## Document Roadmap

This document is the first in a series of 12 explainability files that walk through the entire RETINA-Q pipeline — from raw data to deployed application:

| # | Document | Covers |
|---|---|---|
| 01 | Project Overview (this file) | Vision, architecture, clinical context |
| 02 | Data Acquisition & Datasets | Where the training data comes from |
| 03 | Image Preprocessing Pipeline | How raw images are prepared for models |
| 04 | Quantum Computing Fundamentals | Qubits, circuits, and variational quantum algorithms |
| 05 | OCT Quantum Model — Training & Architecture | 8-qubit VQC design and training results |
| 06 | Fundus Hybrid Model — Training & Architecture | EfficientNet + 4-qubit design and training |
| 07 | U-Net Macular Segmentation | Segmentation architecture and loss design |
| 08 | Backend API & Async Infrastructure | FastAPI, Celery, Redis, PostgreSQL |
| 09 | Frontend & User Experience | Next.js terminal UI, SSE streaming, data visualisation |
| 10 | Explainability & Interpretability | Grad-CAM, feature importance, heatmap generation |
| 11 | Deployment & DevOps | Docker, Electron, Capacitor, CI/CD |
| 12 | Feedback Loop & Continuous Improvement | Doctor feedback, quarantine, retraining strategy |
