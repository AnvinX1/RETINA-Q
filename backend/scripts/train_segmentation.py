"""
U-Net Segmentation Training Script
Trains the U-Net model for macular segmentation.

Usage:
    cd backend
    source .venv/bin/activate
    python scripts/train_segmentation.py

For ODIR-5K (no masks), this script generates pseudo-masks using
green-channel thresholding as a starting point. For proper training,
replace with a dataset that has ground-truth segmentation masks
(e.g., IDRiD or DRIVE).
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from loguru import logger

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Add backend root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.unet_model import UNet, CombinedSegmentationLoss


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────
class SegmentationDataset(Dataset):
    """
    Fundus segmentation dataset.
    If mask_dir is provided, uses ground-truth masks.
    Otherwise, generates pseudo-masks from green channel + CLAHE + Otsu.
    """

    def __init__(self, image_paths: list, mask_dir: Path = None, size: int = 256):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def _generate_pseudo_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate a rough macular region mask using green-channel thresholding."""
        green = image[:, :, 1]  # Green channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(green)

        # Otsu thresholding to find bright regions (optic disc / macula)
        _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations to clean
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            # Return a blank sample
            return torch.zeros(1, self.size, self.size), torch.zeros(1, self.size, self.size)

        # Load or generate mask
        if self.mask_dir and (self.mask_dir / img_path.name).exists():
            mask = cv2.imread(str(self.mask_dir / img_path.name), cv2.IMREAD_GRAYSCALE)
        else:
            mask = self._generate_pseudo_mask(image)

        # Preprocess image: green channel + CLAHE
        green = image[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(green)

        # Resize
        enhanced = cv2.resize(enhanced, (self.size, self.size))
        mask = cv2.resize(mask, (self.size, self.size))

        # Normalize
        img_tensor = torch.from_numpy(enhanced).float().unsqueeze(0) / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0) / 255.0

        return img_tensor, mask_tensor


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx+1}/{len(loader)} — loss: {loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    dice_scores = []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        # Dice score
        pred_binary = (outputs > 0.5).float()
        intersection = (pred_binary * masks).sum(dim=(2, 3))
        union = pred_binary.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
        dice = (2 * intersection + 1) / (union + 1)
        dice_scores.extend(dice.mean(dim=1).cpu().numpy())

    return total_loss / len(loader), np.mean(dice_scores)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train RETINA-Q U-Net Segmentation")
    parser.add_argument("--data-dir", type=str, default="data/odir/preprocessed_images",
                        help="Path to fundus image directory")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Path to ground-truth mask directory (optional)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--output", type=str, default="weights/unet_segmentation.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Gather images ─────────────────────────────────────
    data_dir = Path(args.data_dir)
    image_paths = sorted([
        p for p in data_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    if len(image_paths) == 0:
        logger.error(f"No images found in {data_dir}")
        sys.exit(1)

    if len(image_paths) > args.max_samples:
        logger.info(f"Subsampling to {args.max_samples} images")
        np.random.seed(42)
        indices = np.random.choice(len(image_paths), args.max_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]

    logger.info(f"Total images: {len(image_paths)}")

    if args.mask_dir is None:
        logger.warning("No mask directory provided — using pseudo-masks (green channel thresholding)")

    # Split
    train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

    mask_dir = Path(args.mask_dir) if args.mask_dir else None
    train_dataset = SegmentationDataset(train_paths, mask_dir=mask_dir)
    val_dataset = SegmentationDataset(val_paths, mask_dir=mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Model ─────────────────────────────────────────────
    logger.info("Initializing U-Net")
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = CombinedSegmentationLoss(tversky_alpha=0.7, tversky_beta=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ──────────────────────────────────────────
    best_dice = 0.0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(f"{'='*50}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Train — loss: {train_loss:.4f}")
        logger.info(f"Val   — loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), output_path)
            logger.info(f"✓ Saved best model (Dice: {val_dice:.4f}) → {output_path}")

    logger.info(f"\nBest Dice Score: {best_dice:.4f}")
    logger.info(f"Model saved at: {output_path}")


if __name__ == "__main__":
    main()
