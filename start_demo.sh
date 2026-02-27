#!/bin/bash
# RETINA-Q Production Demo Launcher

echo "======================================"
echo " Starting RETINA-Q Production Demo... "
echo "======================================"

# Paths
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BACKEND_DIR="$DIR/backend"
FRONTEND_DIR="$DIR/frontend"

# Kill existing servers
echo "[INFO] Cleaning up existing processes..."
pkill -f uvicorn
pkill -f "next start"
pkill -f "next dev"
sleep 2

# Start Backend
echo "[INFO] Starting Python FastApi Backend (production mode)..."
cd "$BACKEND_DIR"
# Assuming the virtual environment is in the parent directory or activated by the user
# Launching with 4 workers for production-like concurrency
if [ -d "../venv" ]; then
    source ../venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 > /tmp/retina_backend_demo.log 2>&1 &
BACKEND_PID=$!
echo "[INFO] Backend running on http://127.0.0.1:8000 (PID: $BACKEND_PID)"

# Start Frontend
echo "[INFO] Starting Next.js Frontend (production mode)..."
cd "$FRONTEND_DIR"
# Next.js must be built first
echo "[INFO] Building frontend if necessary..."
if [ ! -d ".next" ]; then
    npm run build
fi
npm run start -p 3000 > /tmp/retina_frontend_demo.log 2>&1 &
FRONTEND_PID=$!
echo "[INFO] Frontend running on http://127.0.0.1:3000 (PID: $FRONTEND_PID)"

echo "======================================"
echo "    RETINA-Q System is Online         "
echo "    Access UI: http://localhost:3000  "
echo "======================================"
echo ""
echo "To shut down the demo, run: pkill -f uvicorn && pkill -f 'next start'"
