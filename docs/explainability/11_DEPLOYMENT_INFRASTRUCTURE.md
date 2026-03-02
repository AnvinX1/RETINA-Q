# 11 — Deployment & DevOps Infrastructure

## Introduction

RETINA-Q is designed for multiple deployment targets: a full Docker stack for server/cloud deployment, an Electron wrapper for desktop distribution, and a Capacitor.js shell for mobile (iOS/Android). This document covers every deployment pathway, the containerisation strategy, the training infrastructure, and the operational concerns of running the system in production.

---

## Docker Compose: The Full Stack

The primary deployment method uses Docker Compose to orchestrate six interconnected services:

```yaml
# docker-compose.yml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    environment:
      DATABASE_URL: postgresql://retinaq:retinaq_secret@postgres:5432/retinaq
      REDIS_URL: redis://redis:6379/0
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/1
    depends_on: [postgres, redis]
    volumes:
      - backend-uploads:/app/uploads
      - backend-feedback:/app/feedback

  celery-worker:
    build: ./backend
    command: celery -A app.celery_app worker --loglevel=info --concurrency=2
    environment: # (same as backend)
    depends_on: [redis, backend]

  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      NEXT_PUBLIC_API_URL: http://localhost:8000

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: retinaq
      POSTGRES_USER: retinaq
      POSTGRES_PASSWORD: retinaq_secret
    volumes:
      - postgres-data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0
    volumes:
      - mlflow-data:/mlflow

volumes:
  postgres-data:
  backend-uploads:
  backend-feedback:
  mlflow-data:
```

### Service Dependencies

```
                  ┌─────────┐
                  │  Redis  │
                  └────┬────┘
                       │
            ┌──────────┼──────────┐
            │          │          │
      ┌─────▼─────┐ ┌─▼──────┐  │
      │  Backend   │ │ Celery │  │
      │  (FastAPI) │ │ Worker │  │
      └─────┬──────┘ └────────┘  │
            │                    │
      ┌─────▼──────┐  ┌────────▼──┐
      │  PostgreSQL │  │  MLflow   │
      └────────────┘  └───────────┘
            
      ┌────────────┐
      │  Frontend   │   (independent, calls Backend API)
      │  (Next.js)  │
      └────────────┘
```

### Startup Sequence

```bash
# One-command deployment
./start_docker_stack.sh

# Which runs:
docker compose up --build -d
```

The startup order (managed by `depends_on`):
1. PostgreSQL and Redis start first
2. Backend starts (creates database tables on startup)
3. Celery worker starts (connects to Redis)
4. Frontend starts (connects to Backend)
5. MLflow starts (independent)

---

## Backend Dockerfile

```dockerfile
FROM python:3.11-slim

# System dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Key decisions**:
- **python:3.11-slim**: Slim base (~120MB) instead of full (~900MB). Includes only essential system libraries.
- **libgl1-mesa-glx, libglib2.0-0**: Required by OpenCV for image processing. Without these, `cv2.imread()` crashes.
- **--no-cache-dir**: Prevents pip from caching downloaded packages, reducing image size.

### Dependencies (requirements.txt)

```
fastapi>=0.109.0
uvicorn>=0.25.0
torch>=2.0.0
torchvision>=0.15.0
pennylane>=0.35.0
efficientnet-pytorch>=0.7.1
opencv-python-headless>=4.8.0
scikit-image>=0.22.0
scikit-learn>=1.3.0
celery>=5.3.0
redis>=5.0.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
loguru>=0.7.0
pillow>=10.0.0
numpy>=1.24.0
scipy>=1.11.0
python-multipart>=0.0.6
mlflow>=2.10.0
```

---

## Frontend Dockerfile

```dockerfile
# Stage 1: Build
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 2: Run
FROM node:18-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

# Copy only the standalone output
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
```

**Multi-stage build benefits**:
- Builder stage (~500MB) has all dev dependencies
- Runner stage (~100MB) has only the production server
- No `node_modules` in the final image

---

## Electron Desktop Application

For standalone desktop distribution, RETINA-Q wraps the web frontend in Electron:

### Architecture

```
┌─────────────────────────────────────────┐
│            Electron Window              │
│  ┌───────────────────────────────────┐  │
│  │                                   │  │
│  │    Next.js Frontend               │  │
│  │    (loaded from localhost:3000     │  │
│  │     or bundled static files)      │  │
│  │                                   │  │
│  └───────────────────────────────────┘  │
│                                         │
│  Main Process (Node.js):                │
│  - Window management                    │
│  - System tray integration              │
│  - Auto-update checks                   │
│  - Backend health monitoring            │
└─────────────────────────────────────────┘
```

### Main Process (electron/main.js)

```javascript
const { app, BrowserWindow, Menu } = require("electron");
const path = require("path");

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
    },
    backgroundColor: "#0a0a0a",  // Match terminal aesthetic
    title: "RETINA-Q Diagnostic System",
  });

  // Load the Next.js frontend
  const FRONTEND_URL = process.env.FRONTEND_URL || "http://localhost:3000";
  mainWindow.loadURL(FRONTEND_URL);
}
```

### Retry Logic

The Electron app handles the case where the backend isn't ready yet:

```javascript
async function waitForBackend(url, maxRetries = 30) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch(`${url}/health`);
      if (response.ok) return true;
    } catch (e) {
      // Backend not ready yet
    }
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  return false;
}
```

### Build Targets

```json
// electron/package.json
{
  "build": {
    "appId": "com.retinaq.desktop",
    "productName": "RETINA-Q",
    "linux": {
      "target": ["AppImage", "deb"]
    },
    "win": {
      "target": ["nsis", "portable"]
    },
    "mac": {
      "target": ["dmg"]
    }
  }
}
```

Electron Builder produces platform-specific installers:
- **Linux**: AppImage (portable) and .deb (Debian/Ubuntu)
- **Windows**: NSIS installer and portable .exe
- **macOS**: .dmg disk image

---

## Capacitor Mobile Deployment

For iOS and Android distribution, the Next.js frontend is wrapped using Capacitor.js:

### Process

1. **Export static HTML** from Next.js:
   ```bash
   next build && next export
   ```

2. **Initialise Capacitor**:
   ```bash
   npx cap init "RETINA-Q" "com.retinaq.mobile"
   npx cap add ios
   npx cap add android
   ```

3. **Copy web assets**:
   ```bash
   npx cap copy
   ```

4. **Build native apps**:
   ```bash
   npx cap open ios      # Opens Xcode
   npx cap open android  # Opens Android Studio
   ```

### Network Configuration

Mobile apps connect to a remote backend (not localhost):

```
NEXT_PUBLIC_API_URL=http://192.168.1.100:8000
```

The CORS configuration already includes `capacitor://localhost` as an allowed origin.

---

## Training Infrastructure

### GPU Cluster

Model training was performed on a remote GPU cluster:

- **Hardware**: Dual NVIDIA L40S GPUs (48 GB VRAM each)
- **OS**: Ubuntu 22.04 LTS
- **CUDA**: 12.1
- **PyTorch**: 2.0+ with CUDA support
- **PennyLane**: 0.35+ with `default.qubit` (state-vector simulation on GPU)

### Remote Access

```bash
# SSH tunnel for Jupyter/monitoring
ssh -L 8888:localhost:8888 user@gpu-cluster

# Or direct script execution
scp train_oct.py user@gpu-cluster:~/retinaq/
ssh user@gpu-cluster "cd retinaq && python train_oct.py"
```

### Training Times

| Model | Epochs | Time/Epoch | Total |
|---|---|---|---|
| OCT (8-qubit, backprop) | 30 | ~7 minutes | ~3.5 hours |
| Fundus (4-qubit hybrid) | 20 | ~12 minutes | ~4 hours |
| U-Net Segmentation | 50 | ~3 minutes | ~2.5 hours |

With `parameter-shift` differentiation (the original method), the OCT model took ~100 minutes per epoch — switching to `backprop` provided a 14× speedup.

---

## MLflow Model Registry

MLflow provides model versioning and experiment tracking:

```python
# services/mlflow_registry.py
import mlflow

class MLflowRegistry:
    def __init__(self, tracking_uri):
        mlflow.set_tracking_uri(tracking_uri)
    
    def log_model(self, model, name, metrics):
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
            mlflow.pytorch.log_model(model, name)
    
    def load_latest(self, name):
        model_uri = f"models:/{name}/latest"
        return mlflow.pytorch.load_model(model_uri)
```

MLflow runs as a Docker service on port 5000 and stores model artifacts in a persistent volume. It is currently **optional** (controlled by `MLFLOW_ENABLED=false`).

---

## Persistent Volumes

Docker volumes ensure data survives container restarts:

| Volume | Purpose | Mount Point |
|---|---|---|
| `postgres-data` | Patient records, scan history | `/var/lib/postgresql/data` |
| `backend-uploads` | Uploaded retinal images | `/app/uploads` |
| `backend-feedback` | Doctor feedback logs, quarantined images | `/app/feedback` |
| `mlflow-data` | Model artifacts, experiment metadata | `/mlflow` |

---

## Environment Variables

The complete set of environment variables across services:

```bash
# Backend
DEBUG=false
HOST=0.0.0.0
PORT=8000
DATABASE_URL=postgresql://retinaq:retinaq_secret@postgres:5432/retinaq
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/1
MLFLOW_ENABLED=false
MLFLOW_TRACKING_URI=http://mlflow:5000
SHADOW_ENABLED=false

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000

# PostgreSQL
POSTGRES_DB=retinaq
POSTGRES_USER=retinaq
POSTGRES_PASSWORD=retinaq_secret

# Electron
FRONTEND_URL=http://localhost:3000
```

---

## Native Development (No Docker)

For development without Docker:

```bash
# Terminal 1: Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Celery Worker
cd backend
celery -A app.celery_app worker --loglevel=info --concurrency=2

# Terminal 3: Frontend
cd frontend
npm install
npm run dev

# Terminal 4: Electron (optional)
cd electron
npm install
npx electron .
```

Requires local PostgreSQL and Redis instances:
```bash
# PostgreSQL
sudo systemctl start postgresql
createdb retinaq

# Redis
sudo systemctl start redis-server
```

Or use the convenience script:
```bash
./start_demo.sh   # Starts backend + frontend natively
```

---

## Health Monitoring

The `/health` endpoint enables load balancers and monitoring tools to check system status:

```json
{
  "status": "healthy",
  "models_loaded": true,
  "database": "connected",
  "redis": "connected",
  "uptime_seconds": 3600,
  "version": "1.0.0"
}
```

In a production deployment, this endpoint would be monitored by:
- Docker health checks
- Kubernetes liveness/readiness probes
- External monitoring services (Datadog, Prometheus + Grafana)

The next document (12) covers the clinician feedback loop and continuous improvement strategy.
