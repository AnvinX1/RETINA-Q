# RETINA-Q

Hybrid Quantum-Classical Multi-Modal Retinal Disease Diagnosis System

RETINA-Q is an AI-powered clinical decision support system that integrates
quantum machine learning with deep learning for automated retinal diagnosis
using OCT and Fundus images.

---

## Core Capabilities

- OCT Binary Classification (Normal vs CSR)
- Fundus Binary Classification (Healthy vs CSCR)
- Conditional Macular Segmentation
- Grad-CAM Explainability
- Feature-Level Quantum Interpretability
- REST API Deployment

---

## Architecture Overview

User Upload  
→ Image Type Detection  
→ OCT Quantum Pipeline OR Fundus Quantum-EfficientNet  
→ Conditional Segmentation  
→ Explainability + Report

---

## Tech Stack

- PyTorch
- PennyLane
- EfficientNet-B0
- U-Net
- FastAPI
- Next.js
- PostgreSQL
- Docker

---

## Performance Targets

- ≥ 92% Classification Accuracy
- ≥ 0.90 Dice Score
- < 2s Inference Time

---

## Folder Structure
