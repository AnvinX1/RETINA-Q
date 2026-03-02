# 12 — Feedback Loop & Continuous Improvement

## Introduction

A model deployed today will degrade over time. Patient demographics shift, imaging equipment gets upgraded, new pathologies emerge, and the data distribution drifts away from what the model was trained on. RETINA-Q addresses this through a **clinician-in-the-loop feedback system** that captures real-world performance data and sets the foundation for continuous model improvement.

This is not an afterthought — it is a core architectural feature, built into every layer from the database schema to the frontend UI.

---

## The Feedback Mechanism

### How It Works

After viewing a prediction, the clinician has two options:

1. **Concur**: "I agree with the model's diagnosis." This confirms the prediction was correct.
2. **Override**: "The model is wrong. The correct diagnosis is [X]." This flags an error and provides the ground truth.

```
┌──────────────────────────────────────────────────────┐
│                PREDICTION RESULT                     │
│                                                      │
│  Diagnosis: CSCR DETECTED                            │
│  Confidence: 87.3%                                   │
│  Heatmap: [image]                                    │
│                                                      │
│  ┌────────────────────┐  ┌─────────────────────────┐ │
│  │  ✓ CONCUR WITH     │  │  ✗ OVERRIDE —           │ │
│  │    DIAGNOSIS        │  │    INCORRECT            │ │
│  └────────────────────┘  └─────────────────────────┘ │
│                                                      │
│  (Override opens correction form:)                   │
│  Correct diagnosis: [Normal ▼]                       │
│  Clinical notes: [________________________]          │
│                                                      │
│  [SUBMIT FEEDBACK]                                   │
└──────────────────────────────────────────────────────┘
```

### API Endpoint

```http
POST /api/feedback
Content-Type: application/json

{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "verdict": "override",
  "correction_label": "Normal",
  "notes": "No subretinal fluid visible. Appears to be normal variant of RPE pigmentation."
}
```

### Backend Processing

When feedback is received:

```python
@router.post("/feedback")
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    # 1. Log to structured JSONL file
    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "job_id": request.job_id,
        "verdict": request.verdict,
        "correction_label": request.correction_label,
        "notes": request.notes
    }
    
    with open("feedback/feedback_log.jsonl", "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")
    
    # 2. Update database record
    scan = db.query(Scan).filter(Scan.job_id == request.job_id).first()
    if scan:
        scan.feedback_status = request.verdict
        scan.feedback_label = request.correction_label
        db.commit()
    
    # 3. If overridden, quarantine the image
    if request.verdict == "override":
        quarantine_image(request.job_id, request.correction_label)
    
    return {"status": "feedback_recorded"}
```

---

## The Quarantine System

When a clinician overrides a prediction, the original uploaded image is moved to a quarantine directory, organised by the corrected label:

```python
def quarantine_image(job_id, correction_label):
    source = f"uploads/{job_id}.jpg"
    target_dir = f"feedback/quarantine/{correction_label}"
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy2(source, f"{target_dir}/{job_id}.jpg")
```

### Quarantine Directory Structure

```
feedback/
├── feedback_log.jsonl          ← Structured log of all feedback
└── quarantine/
    ├── Normal/                 ← Images the model wrongly called CSCR
    │   ├── abc123.jpg
    │   ├── def456.jpg
    │   └── ...
    └── CSR/                    ← Images the model wrongly called Normal
        ├── ghi789.jpg
        └── ...
```

This structure is immediately ready for retraining: `quarantine/Normal/` images become negative examples, `quarantine/CSR/` images become positive examples.

---

## Feedback Data Schema

### JSONL Log Format

Each line in `feedback_log.jsonl` is a standalone JSON object:

```json
{"timestamp": "2026-02-15T14:23:46.789Z", "job_id": "abc123", "verdict": "concur", "correction_label": null, "notes": null}
{"timestamp": "2026-02-15T14:25:12.345Z", "job_id": "def456", "verdict": "override", "correction_label": "Normal", "notes": "Artifact on image border confused the model"}
{"timestamp": "2026-02-15T14:30:00.000Z", "job_id": "ghi789", "verdict": "override", "correction_label": "CSR", "notes": "Subtle fluid pocket in inferior macula, model missed it"}
```

### Database Feedback Fields

The `scans` table tracks feedback status:

| Field | Type | Values |
|---|---|---|
| `feedback_status` | String | `"pending"`, `"concurred"`, `"overridden"` |
| `feedback_label` | String | `"Normal"`, `"CSR"`, or null |

This enables querying:
```sql
-- All model errors
SELECT * FROM scans WHERE feedback_status = 'overridden';

-- Model accuracy from clinical feedback
SELECT 
  COUNT(*) as total_reviewed,
  SUM(CASE WHEN feedback_status = 'concurred' THEN 1 ELSE 0 END) as correct,
  SUM(CASE WHEN feedback_status = 'overridden' THEN 1 ELSE 0 END) as incorrect
FROM scans 
WHERE feedback_status IS NOT NULL;

-- False positive rate
SELECT COUNT(*) FROM scans 
WHERE prediction = 'CSR' AND feedback_label = 'Normal';

-- False negative rate
SELECT COUNT(*) FROM scans 
WHERE prediction = 'Normal' AND feedback_label = 'CSR';
```

---

## Continuous Improvement Strategy

### Phase 1: Monitor (Passive)

In the initial deployment phase, feedback is collected but no automated retraining occurs:

1. **Accumulate feedback data** — build a dataset of clinician-validated predictions
2. **Track key metrics** — accuracy, false positive rate, false negative rate, by scan type
3. **Identify patterns** — are certain image types more error-prone? Do certain clinical sites produce more overrides?

### Phase 2: Analyse (Periodic)

Monthly or quarterly review of accumulated feedback:

```python
# Analysis script
import pandas as pd

feedback = pd.read_json("feedback/feedback_log.jsonl", lines=True)

# Overall accuracy from clinical feedback
total = len(feedback)
correct = len(feedback[feedback.verdict == "concur"])
print(f"Clinical accuracy: {correct/total:.1%} ({correct}/{total})")

# Error breakdown
overrides = feedback[feedback.verdict == "override"]
false_positives = overrides[overrides.correction_label == "Normal"]
false_negatives = overrides[overrides.correction_label == "CSR"]

print(f"False positives: {len(false_positives)} ({len(false_positives)/total:.1%})")
print(f"False negatives: {len(false_negatives)} ({len(false_negatives)/total:.1%})")
```

### Phase 3: Retrain (Active)

Once sufficient quarantined data accumulates (recommended: ≥100 overridden samples per class):

```python
# Retraining with quarantine data
from pathlib import Path

# Gather quarantined images with corrected labels
quarantine_normal = list(Path("feedback/quarantine/Normal").glob("*.jpg"))
quarantine_csr = list(Path("feedback/quarantine/CSR").glob("*.jpg"))

# Add to training set
new_train_data = [
    (img, 0) for img in quarantine_normal  # Corrected: these are Normal
] + [
    (img, 1) for img in quarantine_csr     # Corrected: these are CSR
]

# Fine-tune the model on combined (original + quarantine) data
# Use lower learning rate to avoid catastrophic forgetting
model = load_current_model()
optimizer = Adam(model.parameters(), lr=1e-5)  # 10x lower than original

for epoch in range(5):  # Brief fine-tuning
    for images, labels in combined_loader:
        # ... standard training loop ...
```

### Phase 4: Validate and Deploy

Before deploying a retrained model:

1. **Hold-out validation**: Test on a separate set of clinician-validated images.
2. **A/B testing** (shadow mode): Run the new model in parallel with the current model, comparing predictions without affecting clinical workflow.
3. **MLflow versioning**: Register the new model version in MLflow.
4. **Rollback capability**: Keep the previous model version available for instant rollback.

```python
# Shadow A/B testing (enabled via SHADOW_ENABLED=true)
if settings.shadow_enabled:
    shadow_result = shadow_model.predict(image)
    log_shadow_prediction(job_id, shadow_result, production_result)
```

---

## Feedback Analytics Dashboard (Planned)

Future iterations will include a dedicated analytics view showing:

### Model Performance Over Time

```
Accuracy (from clinical feedback)
100% ┤
 90% ┤          ╭──────────
 80% ┤     ╭────╯
 70% ┤─────╯
 60% ┤
     └──────────────────────
     Jan  Feb  Mar  Apr  May
```

### Error Distribution

```
Top error patterns:
┌─────────────────────────────────────────────┐
│ 1. Low-quality images (32% of overrides)    │
│ 2. Rare pathology variants (28%)            │
│ 3. Bilateral comparison needed (18%)        │
│ 4. Artefacts/reflections (12%)              │
│ 5. Other (10%)                              │
└─────────────────────────────────────────────┘
```

### Clinician Agreement Rate

Track inter-rater reliability between the model and multiple clinicians:
- If Clinician A concurs but Clinician B overrides the same image, the model is in a genuine diagnostic grey zone.
- High inter-clinician disagreement on overrides suggests the model error may actually be acceptable.

---

## Data Governance

### Feedback Data Retention

- **JSONL logs**: Retained indefinitely (append-only, versioned)
- **Quarantined images**: Retained for retraining; purged after successful model update
- **Database records**: Retained as part of patient history (subject to data retention policies)

### Audit Trail

Every prediction and its associated feedback creates an audit trail:

```
Prediction {
  job_id: "abc123",
  timestamp: "2026-02-15T14:23:45Z",
  image_type: "fundus",
  prediction: "CSR",
  confidence: 0.873,
  model_version: "fundus_v1.0"
}
  ↓
Feedback {
  job_id: "abc123",
  timestamp: "2026-02-15T14:25:12Z",
  verdict: "override",
  correction: "Normal",
  clinician_notes: "Artifact on image border"
}
```

This audit trail satisfies:
- Clinical governance requirements
- Regulatory audit needs (FDA, EU AI Act)
- Model debugging and incident investigation

---

## The Virtuous Cycle

The feedback loop creates a virtuous cycle:

```
                    ┌─────────────┐
                    │   Deploy    │
                    │   Model     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Clinical   │
                    │   Usage     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Clinician  │
                    │  Feedback   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Quarantine │
                    │  Errors     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Analyse &  │
                    │  Retrain    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Validate   │
                    │  A/B Test   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Deploy     │
                    │  v2.0       │──────────▶ (cycle repeats)
                    └─────────────┘
```

Each cycle:
- Increases model accuracy on real-world clinical data
- Identifies and eliminates systematic failure modes
- Builds clinician trust through demonstrated improvement
- Creates an ever-growing, clinician-validated dataset

---

## Summary

The feedback loop is what transforms RETINA-Q from a static AI tool into a living, improving diagnostic system. By embedding clinician expertise directly into the model improvement pipeline, we ensure that:

1. **Every error teaches the model** — overridden predictions become future training data
2. **Clinicians retain authority** — the system assists, not replaces, expert judgement
3. **Performance improves continuously** — each deployment cycle builds on accumulated clinical wisdom
4. **Transparency is maintained** — full audit trails for every prediction and feedback event

---

## Series Conclusion

This is the final document in the 12-part RETINA-Q Explainability Content series. Together, these documents cover the complete journey from concept to deployed system:

| # | Document | What It Covers |
|---|---|---|
| 01 | Project Overview | Vision, architecture, clinical context |
| 02 | Data Acquisition | Kermany2018, ODIR-5K, data sourcing |
| 03 | Image Preprocessing | CLAHE, feature extraction, augmentation |
| 04 | Quantum Fundamentals | Qubits, gates, VQCs, PennyLane |
| 05 | OCT Model Training | 8-qubit architecture, training, results |
| 06 | Fundus Model Training | EfficientNet + 4-qubit hybrid, dual pathway |
| 07 | U-Net Segmentation | Encoder-decoder, BCE+Dice+Tversky loss |
| 08 | Backend API | FastAPI, Celery, Redis, PostgreSQL |
| 09 | Frontend Development | Next.js, terminal UI, SSE streaming |
| 10 | Explainability | Grad-CAM, feature importance, heatmaps |
| 11 | Deployment | Docker, Electron, Capacitor, training infra |
| 12 | Feedback Loop | Clinician feedback, quarantine, retraining |

Together, these documents provide a complete, transparent account of every design decision, implementation detail, and engineering trade-off in the RETINA-Q system.
