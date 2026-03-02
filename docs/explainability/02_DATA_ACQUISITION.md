# 02 — Data Acquisition & Datasets

## Introduction

Every machine learning model is only as good as the data it learns from. For RETINA-Q, we needed two distinct types of retinal imagery — OCT (Optical Coherence Tomography) cross-sections and colour fundus photographs — each labelled with ground-truth diagnoses. This document explains exactly where the data came from, how it was obtained, and what preprocessing was applied before it entered the training pipeline.

---

## The Two Imaging Modalities

### OCT (Optical Coherence Tomography)

OCT is a non-invasive imaging technique that uses low-coherence interferometry to produce high-resolution cross-sectional images of the retina. Think of it as an "ultrasound using light" — it reveals the individual layers of the retina (nerve fibre layer, ganglion cell layer, inner/outer nuclear layers, RPE, choroid, etc.) with micrometre-level resolution.

**What CSCR looks like on OCT:**
- Sub-retinal fluid accumulation (dark pockets between the photoreceptor layer and RPE)
- Detachment or irregularity of the retinal pigment epithelium
- Increased central retinal thickness
- In chronic cases, photoreceptor outer segment elongation

### Fundus Photography

A fundus photograph captures a colour image of the back of the eye — the retina, optic disc, macula, and retinal vasculature — through a mydriatic or non-mydriatic fundoscope.

**What CSCR looks like on fundus:**
- Serous elevation of the neurosensory retina (often seen as a subtle circular elevation at the macula)
- Focal areas of leakage (bright spots)
- Altered or mottled pigmentation at the posterior pole
- In chronic cases, gravitational tracks of subretinal fluid

---

## Dataset 1: Kermany2018 (OCT Images)

### Source

The primary OCT dataset is the **Kermany et al. 2018** dataset, one of the most widely used benchmarks in retinal OCT classification research. It was originally published alongside the paper:

> Kermany, D.S., Goldbaum, M., Cai, W. et al. "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning." *Cell*, 172(5), 1122–1131 (2018).

### Acquisition

The dataset was downloaded from **Kaggle** using the Kaggle CLI:

```bash
kaggle datasets download -d paultimothymooney/kermany2018
```

This requires a Kaggle account and API token (stored as `~/.kaggle/kaggle.json`).

### Dataset Composition

The full Kermany2018 dataset contains approximately **84,495 OCT images** across four classes:

| Class | Description | Count (approx.) |
|---|---|---|
| **NORMAL** | Healthy retina | ~26,315 |
| **CNV** | Choroidal Neovascularisation | ~37,206 |
| **DME** | Diabetic Macular Edema | ~11,349 |
| **DRUSEN** | Drusen deposits | ~8,617 |

### How We Used It

For RETINA-Q's binary classification task (Normal vs. CSCR/Abnormal), we restructured the dataset:

- **Class 0 (Normal)**: All images from the `NORMAL` directory
- **Class 1 (Abnormal/CSR)**: Images from `CNV`, `DME`, and `DRUSEN` combined

This gives the model a binary decision boundary: "Is this OCT scan showing a healthy retina or signs of pathology?"

The rationale: while the original dataset doesn't contain a dedicated CSCR label, training on multiple retinal pathologies teaches the quantum classifier to distinguish healthy retinal structure from disrupted layering — and CSCR's subretinal fluid pockets create disruption patterns within this learned feature space.

### Directory Structure After Download

```
data/
  OCT2017/
    train/
      NORMAL/      (≈26K images)
      CNV/         (≈37K images)
      DME/         (≈11K images)
      DRUSEN/      (≈9K images)
    test/
      NORMAL/
      CNV/
      DME/
      DRUSEN/
    val/
      NORMAL/
      CNV/
      DME/
      DRUSEN/
```

### Image Characteristics

- **Format**: JPEG
- **Resolution**: Varies (typically 496 × 512, 496 × 768, or similar; resized to 224 × 224 during preprocessing)
- **Colour**: Greyscale (single channel cross-sections)
- **Quality**: Clinical-grade scans from Heidelberg Spectralis or similar OCT devices

---

## Dataset 2: ODIR-5K (Fundus Images)

### Source

The fundus dataset comes from the **Ocular Disease Intelligent Recognition (ODIR-5K)** challenge dataset, compiled from multiple hospitals and eye centres in China.

### Acquisition

```bash
kaggle datasets download -d andrewmvd/ocular-disease-recognition-odir5k
```

### Dataset Composition

ODIR-5K contains **5,000 patient records** with paired left/right fundus photographs. Each record includes structured annotations:

| Label | Description |
|---|---|
| N | Normal fundus |
| D | Diabetic Retinopathy |
| G | Glaucoma |
| C | Cataract |
| A | Age-related Macular Degeneration |
| H | Hypertensive Retinopathy |
| M | Myopia |
| O | Other diseases/abnormalities |

### How We Used It

For RETINA-Q's binary fundus classification:

- **Class 0 (Healthy)**: Images labelled `N` (Normal)
- **Class 1 (CSCR/Abnormal)**: Images from all other categories

Similar to the OCT approach, training on a broad abnormal class teaches the EfficientNet + quantum hybrid to distinguish healthy retinal appearance from pathological features. The model learns to detect disruptions in the normal fundal pattern — an ability that transfers to CSCR detection.

### Image Characteristics

- **Format**: JPEG
- **Resolution**: Various (512 × 512 to 2592 × 1728; resized to 224 × 224)
- **Colour**: RGB (full colour fundus photographs)
- **Quality**: Mixed — some high-quality clinical images, others with artifacts, over/under-exposure, or low contrast

---

## Data Splitting Strategy

### OCT Dataset

The Kermany2018 dataset comes pre-split:

| Split | Purpose | Approximate Size |
|---|---|---|
| `train/` | Model training | ~83,484 images |
| `val/` | Hyperparameter tuning | ~32 images per class |
| `test/` | Final evaluation | ~968 images |

During training, we used the provided splits directly. The validation set was supplemented with a portion of the training set (typically 10–15%) for more robust validation metrics.

### Fundus Dataset

The ODIR-5K dataset required manual splitting:

```python
from sklearn.model_selection import train_test_split

train_paths, val_paths = train_test_split(
    all_paths, test_size=0.2, random_state=42, stratify=labels
)
```

- **Train**: 80% (≈4,000 images)
- **Validation**: 20% (≈1,000 images)
- Stratified split to preserve class balance across splits.

---

## Data Quality Considerations

### Challenges Encountered

1. **Class Imbalance**: The OCT dataset is heavily skewed toward CNV. We addressed this with weighted random sampling during training:
   ```python
   class_weights = 1.0 / class_counts
   sample_weights = class_weights[labels]
   sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
   ```

2. **Variable Image Quality**: Fundus images from ODIR-5K include low-quality scans with artifacts. Rather than discarding them, we kept them in to improve model robustness — real-world clinical images are often imperfect.

3. **Label Noise**: Multi-label annotations in ODIR-5K can be ambiguous (e.g., "Other" category). We simplified to binary classification to reduce noise impact.

4. **Domain Gap**: Both datasets originate from specific clinical settings and populations. Deployment to different demographics or imaging equipment may require domain adaptation or fine-tuning.

### Data Integrity Measures

- **Deduplication**: Checked for duplicate images across train/val/test splits.
- **Corruption Check**: Verified all images load correctly before training:
  ```python
  from PIL import Image
  for path in all_paths:
      try:
          img = Image.open(path).convert("RGB")
      except Exception:
          corrupted.append(path)
  ```
- **Label Verification**: Spot-checked random samples from each class to confirm label accuracy.

---

## Ethical and Legal Considerations

### Data Licensing

- **Kermany2018**: Released under CC BY 4.0 license. Free to use for research and commercial purposes with attribution.
- **ODIR-5K**: Released for the ODIR challenge. Usage is permitted for research purposes; all patient information has been de-identified.

### Patient Privacy

Both datasets have been fully de-identified before public release:
- No patient names, IDs, or demographic information linked to individual images
- No metadata containing hospital or device serial numbers
- Images stripped of DICOM headers and EXIF data

### Responsible Use

RETINA-Q is intended as a **screening aid**, not a standalone diagnostic tool. All predictions should be reviewed by a qualified ophthalmologist. The feedback loop mechanism (Document 12) ensures clinician oversight remains central to the workflow.

---

## Summary

| Aspect | OCT (Kermany2018) | Fundus (ODIR-5K) |
|---|---|---|
| **Source** | Kaggle (Kermany et al. 2018) | Kaggle (ODIR Challenge) |
| **Total Images** | ~84,495 | ~10,000 (paired L/R) |
| **Classes Used** | Normal vs. Abnormal (binary) | Normal vs. Abnormal (binary) |
| **Image Type** | Greyscale cross-sections | RGB colour photographs |
| **Resolution** | 224 × 224 (resized) | 224 × 224 (resized) |
| **License** | CC BY 4.0 | Research use |
| **Download** | `kaggle datasets download` | `kaggle datasets download` |

The next document (03 — Image Preprocessing Pipeline) explains exactly how these raw images are transformed into model-ready tensors.
