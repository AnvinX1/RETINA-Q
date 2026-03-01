# Research Paper Outline: RETINA-Q

This document is intended to serve as a foundational outline and draft structure for turning the RETINA-Q project into a formal academic research paper or preprint.

## Title Ideas
1.  *Hybrid Quantum-Classical Neural Networks for the Diagnosis of Central Serous Chorioretinopathy via Multi-Modal Retinal Imaging.*
2.  *Overcoming Gradient Bottlenecks in Variational Quantum Circuits for High-Resolution Medical Image Classifiers.*
3.  *RETINA-Q: A Practical Framework for Integrating 8-Qubit Parameterized Quantum Circuits into Clinical Decision Support Pipelines.*

---

## Abstract
**Background:** Quantum Machine Learning (QML) offers theoretical advantages in feature expressivity, but simulating Variational Quantum Circuits (VQCs) in deep learning pipelines remains prohibitively expensive.
**Objective:** Compare the performance of classical CNNs vs. Hybrid Quantum-Classical topologies in diagnosing optical retinal diseases (CSR/CSCR) using OCT and Fundus imagery.
**Methods:** Introduce a hybrid architecture merging an `EfficientNet-B0` feature extractor with a 4-qubit VQC, alongside a standalone 8-qubit quantum classifier for statistical feature maps. Detail the migration from `parameter-shift` to `backprop` state-vector gradients to enable convergence on large multi-thousand-image datasets (Kermany2018, ODIR-5K).
**Results:** [To be populated with final remote GPU metrics. Highlight the AUC/Accuracy thresholds achieved by the quantum models].
**Conclusion:** Highlight whether the quantum layers provided superior feature entanglement and generalization over standard dense classical layers.

---

## 1. Introduction
-   **The Medical Problem:** Diagnosis of retinal diseases (specifically Central Serous Chorioretinopathy) is time-consuming and prone to human error. Multi-modal analysis (OCT + Fundus) is the gold standard.
-   **The Computational Problem:** Standard Deep Learning (DL) is plateauing. Quantum computing offers higher dimensional Hilbert spaces for feature mapping (expressivity), but hardware (NISQ) is noisy, and software simulations are slow.
-   **Proposed Solution:** A hybrid pipeline combining the robust spatial feature extraction of classical CNNs with the high-dimensional entanglement mapping of parameterized quantum circuits.

## 2. Related Work
-   Classical DL in Ophthalmology (ResNets, U-Nets).
-   Early explorations of QML in healthcare (mostly isolated to 2-qubit toy datasets like MNIST).
-   *Gap*: Lack of research running high-qubit-count models natively integrated into end-to-end PyTorch pipelines on large, high-resolution medical imaging datasets.

## 3. Methodology & Architecture
### 3.1 Dataset Preparation
-   Details on preprocessing the **Kermany2018 OCT** dataset (84k images) and the **ODIR-5K Fundus** dataset.
-   Data augmentation strategies (rotations, affine transforms) applied.

### 3.2 The Hybrid Fundus Architecture (EfficientNet-Q)
-   **Classical Backbone:** `EfficientNet-B0` parameters and training paradigm. Why B0 instead of ResNet50? (parameter efficiency).
-   **Dimensionality Bridge:** The dense neural layers bridging $1280 \rightarrow 4$ dimensions.
-   **Quantum Topography:** Detail the 4-qubit circuit structure. 
    -   $R_Y$ angle embedding.
    -   6 variational layers mapped by $R_Y$ and $R_Z$ rotations.
    -   CNOT entanglement mapping.
    -   Pauli-Z expectation measurements.
-   **Fusion:** Concatenating the $4$-dim quantum output with the $4$-dim classical input before final classification.

### 3.3 The OCT Model
-   Detail the 8-qubit PennyLane VQC mapping 64 statistical features.

### 3.4 Gradient Computation Strategies (The "Backprop" Shift)
-   *This is a crucial technical contribution.* 
-   Compare the mathematical complexity of `parameter-shift` rules vs `backprop` (state-vector simulation).
-   Provide metrics showing the exponential degradation of step times using parameter shifts on an 8-qubit system, juxtaposed with the rapid convergence enabled by `default.qubit` + PyTorch autograd.

## 4. Experimental Setup
-   **Hardware:** Details of the remote cluster (e.g., NVIDIA L40S, CUDA versions).
-   **Hyperparameters:** Learning rates (AdamW), Batch Sizes, Epochs. 
-   **Loss Functions:** Detail the custom `BCE + Dice + Tversky` composite loss used in the associated U-Net segmentation pipeline to penalize false negatives.

## 5. Results & Discussion
-   **Classification Metrics:** ROC-AUC, Accuracy, Precision, Recall, F1-Score for both models.
-   **Ablation Study (Crucial):** *Did the quantum layer actually help?* Compare the hybrid `EfficientNet-B0 + 4-qubit VQC` against an architecture where the quantum layer is replaced by a standard classical dense layer of equivalent parameter count.
-   **Explainability:** Showcase the Grad-CAM heatmaps. Does the quantum-hybrid model focus on the correct macular pathologies?

## 6. Conclusion
-   Summary of achievements.
-   Limitations (e.g., Simulated qubits vs. actual hardware execution). 
-   Future Work (e.g., Trapped-Ion hardware execution via Amazon Braket).

---

## 7. References
*Include citations for:*
1.  PennyLane/Xanadu QML frameworks.
2.  Kermany OCT dataset paper.
3.  Relevant QML literature (e.g., Schuld et al., "Quantum Machine Learning in Feature Hilbert Spaces").
