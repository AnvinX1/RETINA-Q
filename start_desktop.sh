#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# RETINA-Q Desktop Launcher
# Starts the Docker stack and opens the Electron desktop shell.
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  RETINA-Q Desktop  —  Starting...    ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"

# ── 1. Start Docker Compose stack ──────────────────────────
echo -e "\n${GREEN}[1/3] Starting Docker stack...${NC}"
sudo docker compose up -d --build

# ── 2. Wait for backend health ─────────────────────────────
echo -e "${GREEN}[2/3] Waiting for backend health check...${NC}"
MAX_WAIT=120
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || true)
    if [ "$STATUS" = "200" ]; then
        echo -e "  ✓ Backend is healthy"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
    echo -e "  Waiting... (${ELAPSED}s)"
done

if [ "$STATUS" != "200" ]; then
    echo "  ✗ Backend did not become healthy after ${MAX_WAIT}s"
    echo "  Check logs: sudo docker compose logs backend"
    exit 1
fi

# Also check frontend
echo -e "  Checking frontend..."
ELAPSED=0
while [ $ELAPSED -lt 60 ]; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 2>/dev/null || true)
    if [ "$STATUS" = "200" ]; then
        echo -e "  ✓ Frontend is ready"
        break
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done

# ── 3. Launch Electron ─────────────────────────────────────
echo -e "${GREEN}[3/3] Launching RETINA-Q Desktop...${NC}\n"

cd electron

# Install Electron deps if needed
if [ ! -d "node_modules" ]; then
    echo "  Installing Electron dependencies..."
    npm install
fi

# Launch
npx electron .

# ── Cleanup (optional) ─────────────────────────────────────
echo -e "\n${CYAN}Electron closed. Docker stack is still running.${NC}"
echo "To stop: sudo docker compose down"
