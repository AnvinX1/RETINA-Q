"""
Backend Tests — Feature extraction and model inference.
"""
import numpy as np
import torch
import pytest


class TestOCTFeatureExtraction:
    """Tests for the OCT feature extraction module."""

    def test_extract_features_shape(self):
        from app.utils.oct_feature_extractor import extract_features

        # Create a dummy grayscale image
        image = np.random.randint(0, 256, (300, 300), dtype=np.uint8)
        features = extract_features(image)

        assert features.shape == (64,), f"Expected 64 features, got {features.shape}"

    def test_extract_features_from_color(self):
        from app.utils.oct_feature_extractor import extract_features

        # Color image should be auto-converted to grayscale
        image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
        features = extract_features(image)

        assert features.shape == (64,)

    def test_gradient_features(self):
        from app.utils.oct_feature_extractor import extract_gradient_features

        image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        features = extract_gradient_features(image)

        assert features.shape == (16,)
        assert not np.any(np.isnan(features))

    def test_histogram_features(self):
        from app.utils.oct_feature_extractor import extract_histogram_features

        image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        features = extract_histogram_features(image)

        assert features.shape == (16,)
        assert not np.any(np.isnan(features))

    def test_lbp_features(self):
        from app.utils.oct_feature_extractor import extract_lbp_features

        image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        features = extract_lbp_features(image)

        assert features.shape == (16,)

    def test_texture_variance_features(self):
        from app.utils.oct_feature_extractor import extract_texture_variance_features

        image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        features = extract_texture_variance_features(image)

        assert features.shape == (8,)

    def test_statistical_moments(self):
        from app.utils.oct_feature_extractor import extract_statistical_moments

        image = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
        features = extract_statistical_moments(image)

        assert features.shape == (8,)
        assert not np.any(np.isnan(features))


class TestQuantumOCTModel:
    """Tests for the Quantum OCT classification model."""

    def test_model_output_shape(self):
        from app.models.quantum_oct_model import QuantumOCTClassifier

        model = QuantumOCTClassifier()
        model.eval()

        x = torch.randn(1, 64)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1), f"Expected (1, 1), got {output.shape}"

    def test_predict_returns_dict(self):
        from app.models.quantum_oct_model import QuantumOCTClassifier

        model = QuantumOCTClassifier()
        x = torch.randn(1, 64)
        result = model.predict(x)

        assert "prediction" in result
        assert "confidence" in result
        assert "probability" in result
        assert result["prediction"] in ("Normal", "CSR")
        assert 0 <= result["confidence"] <= 1
        assert 0 <= result["probability"] <= 1


class TestUNetModel:
    """Tests for the U-Net segmentation model."""

    def test_unet_output_shape(self):
        from app.models.unet_model import UNet

        model = UNet(in_channels=1, out_channels=1)
        model.eval()

        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1, 256, 256), f"Expected (1,1,256,256), got {output.shape}"

    def test_unet_output_range(self):
        from app.models.unet_model import UNet

        model = UNet(in_channels=1, out_channels=1)
        model.eval()

        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            output = model(x)

        # Sigmoid output should be in [0, 1]
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_dice_loss(self):
        from app.models.unet_model import DiceLoss

        loss_fn = DiceLoss()
        pred = torch.sigmoid(torch.randn(1, 1, 64, 64))
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()

        loss = loss_fn(pred, target)
        assert 0 <= loss.item() <= 1

    def test_tversky_loss(self):
        from app.models.unet_model import TverskyLoss

        loss_fn = TverskyLoss(alpha=0.7, beta=0.3)
        pred = torch.sigmoid(torch.randn(1, 1, 64, 64))
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()

        loss = loss_fn(pred, target)
        assert 0 <= loss.item() <= 1


class TestImageProcessing:
    """Tests for image processing utilities."""

    def test_preprocess_segmentation(self):
        from app.utils.image_processing import preprocess_segmentation

        image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        tensor = preprocess_segmentation(image, size=(256, 256))

        assert tensor.shape == (1, 1, 256, 256)
        assert tensor.min() >= 0
        assert tensor.max() <= 1

    def test_postprocess_mask(self):
        from app.utils.image_processing import postprocess_mask

        mask = np.random.rand(256, 256)
        result = postprocess_mask(mask, (300, 400))

        assert result.shape == (300, 400)

    def test_numpy_to_base64(self):
        from app.utils.image_processing import numpy_to_base64

        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        b64 = numpy_to_base64(image)

        assert isinstance(b64, str)
        assert len(b64) > 0
