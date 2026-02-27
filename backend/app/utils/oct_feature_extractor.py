"""
OCT Feature Extractor
Extracts 64 statistical features from grayscale OCT images:
- Gradient magnitude statistics
- Histogram distribution features
- Local Binary Pattern (LBP) features
- Texture variance features
- Statistical moments
"""
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from scipy import stats


def extract_gradient_features(image: np.ndarray) -> np.ndarray:
    """Extract gradient-based features (16 features)."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    features = []
    for region in np.array_split(magnitude, 4):
        features.extend([
            np.mean(region),
            np.std(region),
            np.max(region),
            float(np.percentile(region, 75)),
        ])
    return np.array(features, dtype=np.float32)


def extract_histogram_features(image: np.ndarray) -> np.ndarray:
    """Extract histogram distribution features (16 features)."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)

    features = []
    # Split histogram into 4 quartile bins
    quartiles = np.array_split(hist, 4)
    for q in quartiles:
        features.extend([
            np.sum(q),
            np.mean(q),
            np.std(q),
            float(np.argmax(q)),
        ])
    return np.array(features, dtype=np.float32)


def extract_lbp_features(image: np.ndarray) -> np.ndarray:
    """Extract Local Binary Pattern features (16 features)."""
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")

    # LBP histogram
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    # Pad or truncate to 16 features
    if len(hist) >= 16:
        features = hist[:16]
    else:
        features = np.pad(hist, (0, 16 - len(hist)), mode="constant")

    return features.astype(np.float32)


def extract_texture_variance_features(image: np.ndarray) -> np.ndarray:
    """Extract texture variance features using sliding windows (8 features)."""
    h, w = image.shape
    features = []

    # Split image into 2x4 grid and compute variance in each patch
    for row_patch in np.array_split(image, 2, axis=0):
        for col_patch in np.array_split(row_patch, 4, axis=1):
            features.append(np.var(col_patch))

    return np.array(features, dtype=np.float32)


def extract_statistical_moments(image: np.ndarray) -> np.ndarray:
    """Extract statistical moments (8 features)."""
    flat = image.flatten().astype(np.float64)

    features = [
        np.mean(flat),
        np.std(flat),
        float(stats.skew(flat)),
        float(stats.kurtosis(flat)),
        float(np.median(flat)),
        float(np.percentile(flat, 25)),
        float(np.percentile(flat, 75)),
        float(np.percentile(flat, 75) - np.percentile(flat, 25)),  # IQR
    ]

    return np.array(features, dtype=np.float32)


def extract_features(image: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Extract all 64 features from an OCT image.

    Args:
        image: Input image (grayscale or BGR).
        target_size: Resize dimensions.

    Returns:
        np.ndarray of shape (64,) with all features.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    image = cv2.resize(image, target_size)

    # Extract all feature groups
    gradient = extract_gradient_features(image)      # 16
    histogram = extract_histogram_features(image)     # 16
    lbp = extract_lbp_features(image)                 # 16
    texture = extract_texture_variance_features(image) # 8
    moments = extract_statistical_moments(image)       # 8

    features = np.concatenate([gradient, histogram, lbp, texture, moments])
    assert features.shape == (64,), f"Expected 64 features, got {features.shape[0]}"

    return features
