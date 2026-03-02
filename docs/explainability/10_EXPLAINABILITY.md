# 10 — Explainability & Interpretability

## Introduction

A model that says "CSCR detected, 87% confidence" is useful. A model that says "CSCR detected, 87% confidence — and here's exactly what I'm looking at in the image" is **trustworthy**. In clinical settings, explainability isn't a nice-to-have; it's a prerequisite for adoption. No ophthalmologist will rely on a black-box system that can't justify its decisions.

RETINA-Q implements two complementary explainability techniques:

1. **Grad-CAM** for the Fundus model — highlighting which spatial regions of the image drove the prediction.
2. **Gradient-based Feature Importance** for the OCT model — showing which of the 64 extracted features contributed most to the decision.

This document explains both techniques in detail, how they are generated, and how the outputs are presented to clinicians.

---

## Why Explainability Matters in Medical AI

### Regulatory Requirements

Medical AI systems increasingly require explainability for regulatory approval:
- **FDA**: Section 510(k) submissions for AI-based medical devices must demonstrate transparency in decision-making.
- **EU AI Act**: High-risk AI systems (including medical diagnostics) must provide information about the logic involved.
- **Clinical governance**: Hospitals require that AI tools provide audit trails for every prediction.

### Clinical Trust

Ophthalmologists have decades of training in reading retinal images. They need to verify that the AI is "looking at the right things." If a model correctly identifies CSCR but its heatmap highlights blood vessels instead of subretinal fluid, the clinician will (rightly) distrust it.

### Failure Mode Detection

Explainability reveals when a model is right for wrong reasons. For example:
- A model that looks at image borders (camera artefacts) rather than retinal structures
- A model that uses text annotations burned into the image
- A model that correlates with scan device type rather than pathology

Heatmaps and feature importance make these failure modes visible and correctable.

---

## Technique 1: Grad-CAM (Fundus Model)

### What Is Grad-CAM?

**Gradient-weighted Class Activation Mapping** (Grad-CAM) was introduced by Selvaraju et al. (2017). It computes the gradient of the output class score with respect to the feature maps of a specific convolutional layer, then uses these gradients as weights to produce a spatial heatmap showing which regions of the input image were most important for the prediction.

### How It Works — Step by Step

#### Step 1: Forward Pass

Run the fundus image through the EfficientNet-B0 backbone up to the target layer (`_conv_head`, the final convolutional layer):

```python
# The target layer produces feature maps of shape (batch, 1280, 7, 7)
# Each of the 1280 channels is a different "detector" for visual patterns
# The 7×7 spatial grid corresponds to different regions of the input image
```

#### Step 2: Compute Gradients

Backpropagate the class logit (not the sigmoid output) with respect to the target layer activations:

```python
model.zero_grad()
output = model(input_image)
output.backward()

# gradients: how much each activation contributes to the output
gradients = target_layer.grad  # Shape: (1, 1280, 7, 7)
activations = target_layer.output  # Shape: (1, 1280, 7, 7)
```

#### Step 3: Global Average Pool the Gradients

Average the gradients across the spatial dimensions (7×7) to get one weight per channel:

```python
# α_k = (1/Z) Σ_i Σ_j ∂y/∂A^k_ij
weights = gradients.mean(dim=[2, 3])  # Shape: (1, 1280)
```

Each weight $\alpha_k$ represents the importance of feature map $k$ for the prediction.

#### Step 4: Weighted Combination of Activations

Multiply each feature map by its weight and sum across channels:

```python
# L_Grad-CAM = ReLU(Σ_k α_k · A^k)
cam = torch.zeros(7, 7)
for k in range(1280):
    cam += weights[0, k] * activations[0, k]

cam = F.relu(cam)  # Only positive contributions
```

ReLU ensures we only highlight regions that **positively** contribute to the predicted class (CSCR). Negative contributions (regions that suggest "not CSCR") are zeroed out.

#### Step 5: Normalise and Resize

```python
cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)      # Normalise to [0, 1]
cam = cv2.resize(cam.numpy(), (224, 224))  # Upscale to input dimensions
```

#### Step 6: Apply Colour Map and Overlay

```python
heatmap = cv2.applyColorMap(
    (cam * 255).astype(np.uint8), 
    cv2.COLORMAP_JET
)
# JET: blue (cold/unimportant) → green → yellow → red (hot/important)

overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
```

The final overlay shows the original fundus image with a semi-transparent colour gradient:
- **Red/yellow regions**: High importance — the model focused here for its decision.
- **Blue regions**: Low importance — these areas had little influence.
- **No colour**: Zero contribution (after ReLU).

### What Good Grad-CAM Should Look Like

For CSCR detection on fundus images:

**Correct behaviour**:
- Heatmap focuses on the **macular region** (central fovea)
- Hot spots align with areas of serous detachment or RPE changes
- Cool regions cover the optic disc, peripheral retina, and background

**Warning signs**:
- Heatmap focuses on **image borders** → model may be using artefacts
- Heatmap is **uniformly distributed** → model isn't learning spatial features
- Heatmap focuses on **optic disc** → model may be confusing glaucoma features with CSCR

---

## Technique 2: Feature Importance (OCT Model)

### The Challenge

The OCT model doesn't process raw pixels — it processes a 64-dimensional feature vector. Grad-CAM requires convolutional feature maps, which don't exist in this pipeline. Instead, we use **gradient-based feature importance**.

### How It Works

#### Step 1: Compute Gradients w.r.t. Input Features

```python
features_tensor = torch.tensor(features, requires_grad=True)
output = model(features_tensor)
output.backward()

# The gradient of each feature tells us how much changing that feature
# would change the prediction
feature_importance = features_tensor.grad.abs()  # Shape: (64,)
```

The absolute gradient $|\partial y / \partial x_i|$ for feature $i$ measures how sensitive the prediction is to that feature. Features with large gradients had the most influence.

#### Step 2: Normalise

```python
feature_importance = feature_importance / feature_importance.sum()
```

Normalise to get a probability distribution — each feature's contribution as a fraction of the total.

#### Step 3: Group by Feature Category

```python
groups = {
    "Gradient Features (1-16)":  feature_importance[0:16].sum(),
    "Histogram Features (17-32)": feature_importance[16:32].sum(),
    "LBP Features (33-48)":      feature_importance[32:48].sum(),
    "Texture & Moments (49-64)":  feature_importance[48:64].sum(),
}
```

This gives clinicians an interpretable summary: "The model relied 45% on gradient features, 25% on texture, 20% on histogram distribution, and 10% on LBP patterns."

#### Step 4: Spatial Heatmap from Feature Importance

Even though the OCT model uses extracted features, we can still generate a spatial heatmap:

```python
# Reshape 64 importance values into an 8×8 grid
importance_grid = feature_importance.reshape(8, 8)

# Upscale to image dimensions
heatmap = cv2.resize(importance_grid.numpy(), (224, 224))
heatmap = cv2.applyColorMap(
    (heatmap * 255).astype(np.uint8), 
    cv2.COLORMAP_JET
)

# Overlay on original OCT image
overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
```

The 8×8 grid is a spatial approximation: since the 64 features were computed from spatial sub-regions of the image (quadrant-based gradients, regional texture variance), the reshaped grid roughly maps back to spatial locations.

### Interpreting OCT Feature Importance

| Feature Group | High Importance Suggests |
|---|---|
| **Gradient features** | The model is detecting layer boundary disruptions (subretinal fluid blurs the RPE boundary) |
| **Histogram features** | The model is detecting altered brightness distribution (fluid pockets appear as mid-grey regions) |
| **LBP features** | The model is detecting texture abnormalities (pathological tissue has irregular micro-patterns) |
| **Texture/moments** | The model is detecting statistical distribution shifts (skewness/kurtosis change with fluid) |

---

## Implementation in Code

### Explainability Service

The `explainability.py` service provides two main functions:

```python
# services/explainability.py

def generate_gradcam(model, input_tensor, original_image):
    """
    Generate Grad-CAM heatmap for fundus model.
    
    Args:
        model: QuantumFundusModel (EfficientNet backbone)
        input_tensor: Preprocessed image tensor (3, 224, 224)
        original_image: Original RGB image for overlay
    
    Returns:
        Base64-encoded PNG of heatmap overlay
    """
    # ... implementation as described above ...
    return encode_to_base64(overlay)


def generate_feature_importance(model, features, original_image):
    """
    Generate gradient-based feature importance for OCT model.
    
    Args:
        model: QuantumOCTModel (8-qubit VQC)
        features: 64-dimensional feature vector
        original_image: Original greyscale OCT image for overlay
    
    Returns:
        Tuple of (base64_heatmap, feature_importance_dict)
    """
    # ... implementation as described above ...
    return encode_to_base64(overlay), importance_dict
```

### Integration with Inference

```python
# In the Celery task:
def predict_fundus_task(self, job_id, image_path):
    # ... model loading and preprocessing ...
    
    # Classification
    prediction, confidence = model.predict(preprocessed)
    
    # Explainability (always generated)
    heatmap = generate_gradcam(model, preprocessed, original_image)
    
    # Segmentation (if applicable)
    segmentation = unet_model.predict(seg_preprocessed)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "heatmap": heatmap,          # Grad-CAM output
        "segmentation": segmentation
    }
```

---

## Visual Output Format

Both techniques produce a **Base64-encoded PNG image** that is embedded directly in the API response:

```json
{
  "heatmap": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
}
```

The frontend renders this directly as an `<img>` tag:

```tsx
<img src={result.heatmap} alt="Explainability heatmap" />
```

This design eliminates the need for separate image hosting or file serving — the entire result is self-contained in the JSON response.

---

## Limitations and Honest Assessment

### Grad-CAM Limitations

1. **Low spatial resolution**: The 7×7 feature map of EfficientNet-B0's final layer limits the heatmap resolution. Upscaling introduces smoothing, so the heatmap shows approximate regions, not pixel-precise localisation.

2. **Single-layer analysis**: Grad-CAM only analyses one layer. Different layers capture different abstraction levels — the final layer captures high-level concepts but may miss low-level diagnostic details.

3. **Class-specific**: The heatmap is for the predicted class only. For a CSCR prediction, it shows what supports CSCR — not what argues against it.

### Feature Importance Limitations

1. **Linear approximation**: Gradients measure local sensitivity — they assume the relationship between features and output is approximately linear near the current input. Quantum circuits are highly nonlinear, so this is an approximation.

2. **Spatial mapping is approximate**: Reshaping 64 features to an 8×8 grid assumes spatial correspondence that only partially holds (some features are global statistics, not spatial).

3. **Feature interdependence**: The importance of one feature may depend on the values of others (interaction effects). Simple gradient analysis doesn't capture these interactions.

### Future Improvements

- **Grad-CAM++**: Weighted version that better handles multi-object scenarios.
- **SHAP values**: For OCT features, Shapley values would provide theoretically grounded feature attribution.
- **Layer-wise relevance propagation**: Propagates relevance scores through each layer, including the quantum circuit.
- **Quantum circuit interpretability**: Analyse the quantum state vector at intermediate layers to understand what the circuit "learns."

The next document (11) covers deployment and DevOps — how the entire system is packaged, containerised, and delivered.
