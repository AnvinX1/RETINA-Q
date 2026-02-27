"""
OCT Classifier Training Script
Trains the QuantumOCTClassifier (64 features → 8-qubit quantum circuit)
using the Kermany OCT dataset.

Usage:
    cd backend
    source .venv/bin/activate
    python scripts/train_oct.py

Dataset: Kermany2018 (data/oct/)
    Expected structure after unzip:
    data/oct/OCT2017/
        train/
            NORMAL/
            CNV/
            DME/
            DRUSEN/
        test/
            NORMAL/
            CNV/
            DME/
            DRUSEN/

We map: NORMAL → 0, all others → 1 (CSR/Disease)
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from loguru import logger

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import cv2

# Add backend root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.quantum_oct_model import QuantumOCTClassifier
from app.utils.oct_feature_extractor import extract_features


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class OCTFeatureDataset(Dataset):
    """
    Pre-extracts 64 features from OCT images.
    Caches features in memory for faster training.
    """

    def __init__(self, image_paths: list, labels: list, target_size=(224, 224)):
        self.features = []
        self.labels = []

        logger.info(f"Extracting features from {len(image_paths)} images...")
        for i, (path, label) in enumerate(zip(image_paths, labels)):
            try:
                image = cv2.imread(str(path))
                if image is None:
                    continue
                feat = extract_features(image, target_size=target_size)
                self.features.append(feat)
                self.labels.append(label)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")

            if (i + 1) % 500 == 0:
                logger.info(f"  Processed {i+1}/{len(image_paths)} images")

        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        logger.info(f"Dataset ready: {len(self.features)} samples, {self.features.shape[1]} features")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ──────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for quantum stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        outputs = model(features).view(-1)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return total_loss / len(loader), acc, auc


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────
def gather_oct_images(data_dir: Path):
    """Scan Kermany OCT directory for images and labels."""
    image_paths = []
    labels = []

    # Look for standard Kermany structure
    for split in ["train", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            # Try alternate structures
            for alt in ["Train", "training", "OCT2017/train", "OCT2017/Train"]:
                alt_dir = data_dir / alt
                if alt_dir.exists():
                    split_dir = alt_dir
                    break

        if not split_dir.exists():
            continue

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name.upper()
            label = 0 if class_name == "NORMAL" else 1

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpeg", ".jpg", ".png", ".tiff"):
                    image_paths.append(img_path)
                    labels.append(label)

    return image_paths, labels


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train RETINA-Q OCT Classifier")
    parser.add_argument("--data-dir", type=str, default="data/oct",
                        help="Path to OCT dataset directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples to use (for faster training with quantum circuits)")
    parser.add_argument("--output", type=str, default="weights/oct_quantum.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Gather images ─────────────────────────────────────
    data_dir = Path(args.data_dir)
    image_paths, labels = gather_oct_images(data_dir)

    if len(image_paths) == 0:
        logger.error(f"No images found in {data_dir}. Check the directory structure.")
        logger.error("Expected: data/oct/train/NORMAL/, data/oct/train/CNV/, etc.")
        sys.exit(1)

    logger.info(f"Found {len(image_paths)} images")
    logger.info(f"Normal: {labels.count(0)}, Disease: {labels.count(1)}")

    # Subsample if too many (quantum circuits are slow)
    if len(image_paths) > args.max_samples:
        logger.info(f"Subsampling to {args.max_samples} images for quantum training speed")
        indices = np.random.RandomState(42).choice(len(image_paths), args.max_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]
        labels = [labels[i] for i in indices]

    # Train/Val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # ── Create datasets (pre-extract features) ────────────
    train_dataset = OCTFeatureDataset(train_paths, train_labels)
    val_dataset = OCTFeatureDataset(val_paths, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Model ─────────────────────────────────────────────
    logger.info("Initializing QuantumOCTClassifier (8-qubit)")
    model = QuantumOCTClassifier().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # ── Training ──────────────────────────────────────────
    best_auc = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*50}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Train — loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        logger.info(f"Val   — loss: {val_loss:.4f}, acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), output_path)
            logger.info(f"✓ Saved best model (AUC: {val_auc:.4f}) → {output_path}")

    # ── Final evaluation ──────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(output_path, map_location=device, weights_only=True))
    val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)
    logger.info(f"Final — Accuracy: {val_acc:.4f}, AUC: {val_auc:.4f}")
    logger.info(f"Model saved at: {output_path}")


if __name__ == "__main__":
    main()
