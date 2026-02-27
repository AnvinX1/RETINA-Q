"""
U-Net Segmentation Model
For macular segmentation from fundus images.

Architecture: Standard U-Net with encoder-decoder + skip connections.
Loss: BCE + Dice + Tversky (α=0.7, β=0.3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block: Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for single-channel input (green channel CLAHE) to single-channel mask output.
    4 encoder levels, bottleneck, 4 decoder levels with skip connections.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        # Output
        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, H, W) — preprocessed green-channel CLAHE image.
        Returns:
            (batch, 1, H, W) — sigmoid-activated segmentation mask.
        """
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder path with skip connections
        d4 = self.up4(b)
        d4 = self._pad_and_cat(d4, e4)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = self._pad_and_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = self._pad_and_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = self._pad_and_cat(d1, e1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.out_conv(d1))

    @staticmethod
    def _pad_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Pad upsampled tensor to match skip connection dimensions, then concatenate."""
        diff_h = skip.size(2) - upsampled.size(2)
        diff_w = skip.size(3) - upsampled.size(3)
        upsampled = F.pad(
            upsampled,
            [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2],
        )
        return torch.cat([skip, upsampled], dim=1)


# ──────────────────────────────────────────────────────────────
# Loss Functions
# ──────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class TverskyLoss(nn.Module):
    """
    Tversky loss — generalisation of Dice that controls FP/FN weighting.
    α=0.7, β=0.3 → penalise false negatives more (good for medical imaging).
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        tp = (pred_flat * target_flat).sum()
        fp = ((1 - target_flat) * pred_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        return 1 - tversky


class CombinedSegmentationLoss(nn.Module):
    """Combined BCE + Dice + Tversky loss."""

    def __init__(
        self,
        bce_weight: float = 0.3,
        dice_weight: float = 0.3,
        tversky_weight: float = 0.4,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight

        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            self.bce_weight * self.bce(pred, target)
            + self.dice_weight * self.dice(pred, target)
            + self.tversky_weight * self.tversky(pred, target)
        )
