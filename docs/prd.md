# RETINA-Q PRD (Product Requirements Document)

## 1. Product Vision
To build a hybrid quantum-classical medical AI system capable of automated, explainable retinal disease diagnosis directly from OCT (Optical Coherence Tomography) and Fundus photography.

## 2. Target Audience
- **Ophthalmologists & Optometrists**: Requiring second-opinion triage tools for high-volume clinics.
- **Medical Researchers**: Exploring the intersection of Quantum Machine Learning (QML) and medical imaging.

## 3. Core Features

### 3.1 Multi-Modal Diagnosis
- **OCT Classification**: Must accurately distinguish between Normal and CSR (Central Serous Retinopathy) scans using Quantum circuits.
- **Fundus Classification**: Must accurately distinguish between Healthy and CSCR (Central Serous Chorioretinopathy) leveraging deep learning backbones.

### 3.2 Automated Segmentation
- **Macular Region Masking**: The system must automatically highlight and extract the macular region from Fundus images using a trained U-Net architecture.

### 3.3 Explainable AI (XAI)
- **Grad-CAM Mappings**: Classifications must not be a "black box". The system must output heatmap overlays pinpointing exactly which pixels the AI utilized to make its diagnosis.

### 3.4 Modern User Interface
- **Minimalist Dashboard**: An ultra-clean, black-and-white, zero-latency interface built on Shadcn/UI principles.
- **Cross-Platform**: Must function flawlessly on Desktop browsers and native mobile applications (via Ionic Capacitor).

## 4. Non-Functional Requirements (NFR)
- **Latency**: Backend inference for a single image must take `< 2.0 seconds` on a standard CPU.
- **Accuracy Targets**: OCT (>92%), Fundus (>93%), Segmentation Dice Score (>0.90).
- **Extensibility**: The Python inference engine must allow easy hot-swapping of new `.pth` weights.
