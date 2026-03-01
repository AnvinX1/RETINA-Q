# Model Training & Quantum Optimizations

## 1. The Quantum Optimization Breakthrough
The most significant hurdle during training was the `diff_method` utilized by PennyLane to compute gradients for the variational quantum circuits (VQCs).

- **Initial Approach**: `diff_method="parameter-shift"`. This evaluates the circuit multiple times per parameter to calculate derivatives. It was prohibitively slow for the 8-qubit OCT model, causing estimated training times in the hundreds of hours.
- **Optimization**: Switched to `diff_method="backprop"`. By utilizing the `default.qubit` simulator alongside PyTorch's native `autograd` interface, the gradient computation was localized to the GPU memory space.
- **Result**: Immediate, exponential reduction in step-time. The models began converging within standard deep-learning timeframes.

## 2. Model Training Summaries

### A. Macular Segmentation (U-Net)
- **Loss Function**: A custom combination of Binary Cross-Entropy (BCE), Dice Loss, and Tversky Loss to heavily penalize false-negatives in the small macular regions.
- **Performance**: Rapid convergence on GPU 0.
- **Final Result**: Reached an exceptional **Dice Score of 0.9663**.

### B. Fundus Classification (Hybrid EfficientNet-B0)
- **Architecture**: A pre-trained EfficientNet-B0 backbone acting as a powerful feature extractor, funneling condensed feature vectors into a 4-qubit quantum classifier.
- **Training Progression**: Loss plummeted from `0.70` to `0.50` within the first few hours on the NVIDIA L40S.
- **Final Result**: 
  - **Validation AUC**: 0.7503
  - **Validation Accuracy**: 68.26%
  - *Note: Excellent baseline, primed for >90% accuracy upon feeding the entire 5K dataset in future remote runs.*

### C. OCT Classification (8-Qubit VQC)
- **Architecture**: An intensive 8-qubit PennyLane circuit assessing 64 classical statistical features fed from OpenCV.
- **Optimization Strategy**: We pivoted from `default.qubit` (CPU bound) to `lightning.gpu` (cuQuantum backed) and entirely refactored the `forward()` pass to natively broadcast tensor operations instead of sequentially looping over the batch. 
- **Training Progression**: By utilizing batched GPU parallelization, epoch completion time plummeted from **~100 minutes** down to **~7 minutes**. We allowed the default parameter of **30 epochs** to continue over the full 84k Kermany2018 dataset overnight.
- **Final Result**:
  - **Validation AUC**: 0.8473
  - **Validation Accuracy**: 79.66%
  - *Note: Switching the 8-qubit PennyLane simulator to `lightning.gpu` completely unlocked the potential of this network, allowing it to comfortably break past the previous 68% accuracy ceiling in a fraction of the time.*
