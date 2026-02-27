"""
Fundus Classifier Training Script
Trains the QuantumFundusClassifier (EfficientNet-B0 + 4-qubit quantum layer)
using the ODIR-5K dataset.

Usage:
    cd backend
    source .venv/bin/activate
    python scripts/train_fundus.py

Dataset: ODIR-5K (data/odir/)
    - full_df.csv — labels (N=Normal, D, G, C, A, H, M, O=Other)
    - preprocessed_images/ — fundus images

We map: N → Healthy (0), all others → Disease (1)
This gives a binary classifier suitable for the RETINA-Q Healthy vs CSCR task.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from loguru import logger

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Add backend root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.quantum_fundus_model import QuantumFundusClassifier


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class FundusDataset(Dataset):
    """Binary fundus classification dataset from ODIR-5K."""

    def __init__(self, df: pd.DataFrame, image_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["filename"]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(row["binary_label"], dtype=torch.float32)
        return image, label


# ──────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for quantum stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(loader)} — loss: {loss.item():.4f}")

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images).view(-1)
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
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train RETINA-Q Fundus Classifier")
    parser.add_argument("--data-dir", type=str, default="data/odir",
                        help="Path to ODIR-5K dataset directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="weights/fundus_quantum.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load and prepare data ─────────────────────────────
    data_dir = Path(args.data_dir)
    csv_path = data_dir / "full_df.csv"
    image_dir = data_dir / "preprocessed_images"

    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)

    # Binary label: N=0 (Healthy), everything else=1 (Disease)
    df["binary_label"] = (df["N"] == 0).astype(int)

    # Filter to images that exist
    df["img_exists"] = df["filename"].apply(lambda f: (image_dir / f).exists())
    df = df[df["img_exists"]].copy()
    logger.info(f"Total images: {len(df)}")
    logger.info(f"Healthy: {(df['binary_label']==0).sum()}, Disease: {(df['binary_label']==1).sum()}")

    # Train/Val split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["binary_label"], random_state=42)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_dataset = FundusDataset(train_df, image_dir, transform=train_transform)
    val_dataset = FundusDataset(val_df, image_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────
    logger.info("Initializing QuantumFundusClassifier (EfficientNet-B0 + 4-qubit)")
    model = QuantumFundusClassifier(pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
