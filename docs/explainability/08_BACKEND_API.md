# 08 — Backend API & Async Infrastructure

## Introduction

The backend is the orchestration layer that connects everything: it receives image uploads from the frontend, dispatches inference tasks to Celery workers, streams real-time progress updates, manages patient records, and logs doctor feedback. Built on **FastAPI** with **Celery** + **Redis** for asynchronous processing and **PostgreSQL** for persistent storage, it provides a production-ready REST API for the entire diagnostic pipeline.

This document explains every component of the backend architecture.

---

## Technology Choices

### Why FastAPI?

FastAPI was chosen over Flask/Django for several reasons:

1. **Native async support**: Built on top of Starlette, FastAPI handles `async/await` natively, enabling SSE (Server-Sent Events) for real-time streaming.
2. **Automatic API documentation**: Swagger UI and ReDoc are generated automatically from type hints.
3. **Pydantic v2 integration**: Request/response schemas are validated automatically with detailed error messages.
4. **Performance**: One of the fastest Python web frameworks, built on ASGI (Uvicorn).

### Why Celery + Redis?

Model inference (especially with quantum circuits) takes 10–30 seconds per image. Running this on the HTTP request thread would block all other requests. Celery dispatches inference tasks to background workers:

- **Celery**: Distributed task queue supporting retries, rate limiting, and concurrent workers.
- **Redis**: Lightweight message broker and result backend. Port 6379 (DB 0 for broker, DB 1 for results).

### Why PostgreSQL?

Patient records, scan history, and feedback logs need ACID-compliant persistent storage. PostgreSQL 16 provides:
- Relational integrity (patient → scans foreign key)
- JSON column support for flexible metadata
- Mature ecosystem with SQLAlchemy ORM

---

## Application Entry Point

The FastAPI application is defined in `main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="RETINA-Q API",
    description="Hybrid Quantum-Classical Retinal Diagnostic System",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",        # Next.js dev
        "http://localhost:3001",        # Alternative port
        "capacitor://localhost",        # Mobile (Capacitor)
        "http://localhost",             # Electron
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(predict_router, prefix="/api")
app.include_router(segment_router, prefix="/api")
app.include_router(feedback_router, prefix="/api")
app.include_router(patients_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")

# Database initialisation on startup
@app.on_event("startup")
async def startup_event():
    init_db()
```

### CORS Configuration

Cross-Origin Resource Sharing is configured to allow requests from:
- The Next.js frontend (localhost:3000)
- Electron desktop app (localhost)
- Capacitor mobile app (capacitor://localhost)

This is essential because the frontend (port 3000) and backend (port 8000) run on different origins.

---

## API Endpoints in Detail

### POST `/api/predict/oct`

Accepts an OCT image and returns a binary classification.

```
Request:
  - file: UploadFile (JPEG/PNG)
  - async_mode: bool (optional, default=true)

Sync Response (async_mode=false):
  {
    "prediction": "Normal" | "CSR",
    "confidence": 0.87,
    "heatmap": "data:image/png;base64,...",
    "features": { "gradient_mean": 0.42, ... }
  }

Async Response (async_mode=true):
  HTTP 202 Accepted
  {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "pending"
  }
```

**Flow**:
1. Validate file type (JPEG, PNG)
2. Generate UUID job_id
3. Save image to `uploads/{job_id}.{ext}`
4. Dispatch `predict_oct_task.delay(job_id, image_path)` to Celery
5. Return job_id immediately

### POST `/api/predict/fundus`

Accepts a colour fundus image. Returns classification + optional segmentation.

```
Request:
  - file: UploadFile (JPEG/PNG)
  - async_mode: bool (optional, default=true)
  - include_segmentation: bool (optional, default=true)

Async Response:
  HTTP 202 Accepted
  { "job_id": "...", "status": "pending" }
```

### POST `/api/segment`

Standalone segmentation endpoint (no classification).

```
Request:
  - file: UploadFile (JPEG/PNG)

Response:
  {
    "segmentation_mask": "data:image/png;base64,...",
    "segmentation_overlay": "data:image/png;base64,..."
  }
```

### GET `/api/jobs/{job_id}`

Poll a submitted job's status.

```
Response (pending):
  { "job_id": "...", "status": "pending" }

Response (complete):
  {
    "job_id": "...",
    "status": "complete",
    "result": {
      "prediction": "CSR",
      "confidence": 0.92,
      "heatmap": "data:image/png;base64,...",
      "segmentation": "data:image/png;base64,..."
    }
  }

Response (failed):
  {
    "job_id": "...",
    "status": "failed",
    "error": "Model loading failed: ..."
  }
```

### GET `/api/jobs/{job_id}/stream`

Server-Sent Events (SSE) endpoint for real-time progress updates.

```
Event Stream:
  data: {"status": "processing", "step": "loading_model", "progress": 10}
  data: {"status": "processing", "step": "preprocessing", "progress": 25}
  data: {"status": "processing", "step": "quantum_inference", "progress": 50}
  data: {"status": "processing", "step": "explainability", "progress": 75}
  data: {"status": "complete", "progress": 100, "result": {...}}
```

SSE uses a long-lived HTTP connection. The frontend opens an `EventSource` to this endpoint and receives incremental updates as the worker progresses through the inference pipeline.

### POST `/api/feedback`

Clinician feedback on a prediction.

```
Request:
  {
    "job_id": "...",
    "verdict": "concur" | "override",
    "correction_label": "Normal" | "CSR",  // only if verdict=override
    "notes": "Patient has confirmed CSCR history"
  }
```

### GET/POST `/api/patients`

CRUD operations for patient records.

```
POST (create):
  {
    "patient_id": "P-001",
    "name": "John Doe",
    "age": 45,
    "gender": "male",
    "medical_history": "Type 2 diabetes, hypertension"
  }

GET (list):
  /api/patients?page=1&limit=20
```

### GET `/api/scans`

Query scan history with filters.

```
GET /api/scans?patient_id=P-001&image_type=fundus&prediction=CSR
```

### GET `/health`

System health check.

```
Response:
  {
    "status": "healthy",
    "models_loaded": true,
    "database": "connected",
    "redis": "connected"
  }
```

---

## Celery Task System

### Celery Configuration

```python
# celery_app.py
from celery import Celery

celery_app = Celery(
    "retinaq",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1"
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_concurrency=2,
)
```

- **Broker (DB 0)**: Queue of pending tasks.
- **Backend (DB 1)**: Storage for task results.
- **Concurrency = 2**: Two inference tasks can run simultaneously per worker.

### Task Definitions

```python
# tasks.py
@celery_app.task(bind=True, name="predict_oct")
def predict_oct_task(self, job_id, image_path):
    self.update_state(state="PROCESSING", meta={"step": "loading_model"})
    
    # Load model (lazy singleton)
    model_manager = ModelManager()
    
    self.update_state(state="PROCESSING", meta={"step": "preprocessing"})
    features = extract_oct_features(image_path)
    
    self.update_state(state="PROCESSING", meta={"step": "quantum_inference"})
    prediction, confidence = model_manager.predict_oct(features)
    
    self.update_state(state="PROCESSING", meta={"step": "explainability"})
    heatmap = generate_oct_heatmap(features, model_manager.oct_model)
    
    return {
        "prediction": prediction,
        "confidence": confidence,
        "heatmap": heatmap
    }
```

### Task Lifecycle

```
1. API receives upload
2. Task dispatched to Redis broker (state: PENDING)
3. Worker picks up task (state: STARTED)
4. Worker updates progress (state: PROCESSING, meta: {step: ...})
5. Worker completes (state: SUCCESS, result: {...})
   OR Worker fails (state: FAILURE, error: "...")
6. Result stored in Redis backend (TTL: 1 hour)
7. Client polls or streams result
```

---

## Database Layer

### ORM Models

**Patient**:
```python
class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    age = Column(Integer)
    gender = Column(String)
    medical_history = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    scans = relationship("Scan", back_populates="patient", cascade="all, delete-orphan")
```

**Scan**:
```python
class Scan(Base):
    __tablename__ = "scans"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, unique=True, nullable=False, index=True)
    patient_id = Column(String, ForeignKey("patients.patient_id"))
    image_type = Column(String)       # "oct" or "fundus"
    prediction = Column(String)        # "Normal" or "CSR"
    confidence = Column(Float)
    feedback_status = Column(String)   # "pending", "concurred", "overridden"
    feedback_label = Column(String)    # correction label if overridden
    created_at = Column(DateTime, default=datetime.utcnow)
    
    patient = relationship("Patient", back_populates="scans")
```

### Database Sessions

```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

FastAPI's dependency injection provides database sessions to route handlers:

```python
@router.get("/patients")
def list_patients(db: Session = Depends(get_db)):
    return db.query(Patient).all()
```

---

## Configuration Management

All settings are managed through environment variables with Pydantic BaseSettings:

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Database
    database_url: str = "postgresql://retinaq:retinaq_secret@localhost:5432/retinaq"
    
    # Redis
    redis_url: str = "redis://localhost:6379/0"
    
    # Model paths
    oct_model_path: str = "weights/oct_quantum.pth"
    fundus_model_path: str = "weights/fundus_quantum.pth"
    unet_model_path: str = "weights/unet_segmentation.pth"
    
    # Image settings
    image_size: int = 224
    segmentation_size: int = 256
    
    # MLflow
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "http://mlflow:5000"
    
    class Config:
        env_file = ".env"
```

This allows different configurations for development, Docker, and production by simply changing environment variables.

---

## Logging

RETINA-Q uses **Loguru** for structured logging:

```python
from loguru import logger

logger.add(
    "logs/retinaq.log",
    rotation="10 MB",       # New file after 10 MB
    retention="7 days",     # Delete old logs after 7 days
    compression="zip",      # Compress rotated logs
    level="INFO"
)
```

Log entries include timestamps, log levels, module names, and structured data:

```
2026-02-15 14:23:45.123 | INFO | predict:predict_oct:42 - OCT prediction: CSR (confidence: 0.87)
2026-02-15 14:23:46.789 | INFO | feedback:submit:28 - Feedback received: job_id=abc123, verdict=concur
```

---

## Model Management: Lazy Singleton

Models are loaded on first use and cached for subsequent requests:

```python
class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
        return cls._instance
    
    def get_oct_model(self):
        if "oct" not in self._models:
            model = QuantumOCTModel(n_qubits=8, n_layers=8)
            model.load_state_dict(torch.load(settings.oct_model_path))
            model.eval()
            self._models["oct"] = model
        return self._models["oct"]
```

This avoids loading 50+ MB of model weights on every request. The singleton pattern ensures all workers share the same model instance within a process.

---

## Request/Response Schemas

Pydantic models enforce type safety and automatic documentation:

```python
# schemas/responses.py
class PredictionResponse(BaseModel):
    job_id: str
    status: str
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    heatmap: Optional[str] = None
    segmentation: Optional[str] = None

class FeedbackRequest(BaseModel):
    job_id: str
    verdict: Literal["concur", "override"]
    correction_label: Optional[str] = None
    notes: Optional[str] = None

class PatientCreate(BaseModel):
    patient_id: str
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    medical_history: Optional[str] = None
```

---

## Error Handling

The API uses FastAPI's exception handling for consistent error responses:

```python
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )
```

Common error scenarios:
- **400**: Invalid file type, missing required fields
- **404**: Job ID not found, patient not found
- **500**: Model loading failure, inference error
- **503**: Redis/database connection failure

The next document (09) covers how the frontend consumes these APIs to deliver the diagnostic experience.
