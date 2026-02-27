# Training Infrastructure & Remote Deployment

## 1. Local Limitations
Initial testing of the hybrid quantum-classical models on local hardware highlighted severe bottlenecks. The PennyLane quantum circuit simulations, specifically when computing gradients, are computationally expensive and strictly bottlenecked by CPU performance when run without GPU acceleration.

## 2. Remote GPU Server Scaling
To accurately train the models across thousands of epochs, the codebase was `rsync`'d to a remote high-performance computing (HPC) environment.

**Remote Hardware Specs:**
- **Host IP**: `172.16.132.12`
- **GPUs**: Dual NVIDIA L40S GPUs
- **VRAM**: 46 GB per GPU (92 GB Total)

## 3. Distributed Training Strategy
To maximize the utilization of the dual L40S setup, we parallelized the training scripts by binding them to specific CUDA device IDs:

- **GPU 0 (`cuda:0`)**: 
  - Hosted the `train_fundus.py` (EfficientNet-B0 + 4-qubit) workload.
  - Hosted the `train_segmentation.py` (U-Net) workload.
- **GPU 1 (`cuda:1`)**:
  - Hosted the `train_oct.py` (8-qubit quantum classifier) workload, ensuring the intensive PennyLane circuit evaluations did not choke the deep learning backbones on GPU 0.

## 4. Checkpoint Synchronization
Upon the completion of the remote training routines, the compiled PyTorch state dictionaries (`.pth` weights) were synchronized back to the local `backend/weights/` directory for local inference and verification by the Next.js frontend.
