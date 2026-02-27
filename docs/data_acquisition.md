# Dataset Acquisition and Preparation

## 1. Overview
The RETINA-Q system requires substantial retinal imaging data to train the hybrid quantum-classical models. We utilized two primary open-source datasets sourced from Kaggle, totaling over 12.4GB of data.

## 2. Datasets Used

### A. Optical Coherence Tomography (OCT)
- **Source**: `paultimothymooney/kermany2018`
- **Volume**: Large-scale dataset of OCT retinal images.
- **Classes**: Normal vs. CSR (and others). For our binary classification circuit, we focused on learning the differentiating features of the healthy retina versus serous retinopathy.
- **Role**: Feeds into the `oct_feature_extractor.py`, which generates 64 statistical features before passing through the 8-qubit pennyLane circuit.

### B. Ocular Disease Intelligent Recognition (ODIR-5K)
- **Source**: `andrewmvd/ocular-disease-recognition-odir5k`
- **Volume**: 5,000 patients with age, color fundus photography from left and right eyes.
- **Role**: Used to train the `EfficientNet-B0` deep learning backbone combined with the 4-qubit quantum layer. Masking pipelines evaluate the macular regions in these colored photographs.

## 3. Automation & Ingestion
To facilitate seamless deployment to remote servers, we built `backend/scripts/download_datasets.py`.
- **Kaggle API**: The script authenticates via `~/.kaggle/kaggle.json`.
- **Extraction**: It automatically downloads, extracts the heavy ZIP archives, and structures them into `backend/data/oct/` and `backend/data/odir/`.
- **Local Cleanup**: After extracting features and achieving model convergence, the bulk raw image data was purged from the local git repository (and `.gitignore` updated) to prevent repository bloat, while retaining a reminder (`TODO_IMPROVEMENTS.md`) to re-sync the full dataset for future accuracy scaling.
