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
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import cv2

# Add backend root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.quantum_oct_model import QuantumOCTClassifier
from app.utils.oct_feature_extractor import extract_features


# ──────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class OCTFeatureDataset(Dataset):
    """
    Reads images from disk, applies stochastic transforms,
    and extracts 64 statistical features on the fly.
    """

    def __init__(self, image_paths: list, labels: list, target_size=(224, 224), transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.target_size = target_size
        self.transform = transform
        logger.info(f"Initialized dataset with {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = cv2.imread(str(path))
        
        # Fallback for corrupted images
        if image is None:
            image = np.zeros((*self.target_size, 3), dtype=np.uint8)

        # Apply augmentation
        if self.transform:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(image_rgb)
            pil_img = self.transform(pil_img)
            image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Extract 64 classical features
        feat = extract_features(image, target_size=self.target_size)

        return (
            torch.tensor(feat, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )



# ──────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, (features, labels) in enumerate(loader):
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

        # Log every batch due to high quantum simulation time
        logger.info(f"  Batch {batch_idx+1}/{len(loader)} — loss: {loss.item():.4f}")

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

    logger.info(f"Found {len(image_paths)} images")
    logger.info(f"Normal: {labels.count(0)}, Disease: {labels.count(1)}")

    # Train/Val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # ── Create datasets (on-the-fly extraction) ───────────
    train_dataset = OCTFeatureDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = OCTFeatureDataset(val_paths, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

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
