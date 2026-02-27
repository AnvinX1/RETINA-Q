# Future Improvements & TODOs

## 1. Scale Up Training Data for Higher Accuracy
**Reminder:** We are currently only using a subset of the datasets (e.g., 5,000 images out of 84,000 for OCT Kermany) to ensure the 8-qubit quantum models train in a reasonable timeframe. 

To improve the `0.53` AUC of the OCT model to clinical levels:
- **Use the full dataset:** Run the feature extraction pipeline on all 84,484 OCT images.
- **Increase Quantum Dept:** Expand the PennyLane circuit from 4 layers to 8+ layers.
- **Data Augmentation:** Apply random rotations, flips, and color jitter to the OCT and ODIR-5K fundus datasets during DataLoader initialization.

*Note: The full 12.4GB datasets are safely stored on the remote GPU server (`172.16.132.12`) under `~/retina-q/backend/data/`. They have been removed from the local machine to save disk space.*
