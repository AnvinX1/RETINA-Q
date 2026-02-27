#!/bin/bash
# RETINA-Q Docker Automation Script

echo "=========================================="
echo " Starting RETINA-Q Stack System (Docker)  "
echo "=========================================="

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed or not in PATH."
    echo "Please install Docker Desktop and try again."
    exit 1
fi

# Ensure docker compose is available
if docker compose version &> /dev/null; then
    COMPOSE_CMD=("docker" "compose")
elif docker-compose version &> /dev/null; then
    COMPOSE_CMD=("docker-compose")
else
    echo "[ERROR] Docker Compose plugin is not installed."
    exit 1
fi

echo "[INFO] Using compose command: ${COMPOSE_CMD[*]}"
echo "[INFO] Tearing down any old containers..."
"${COMPOSE_CMD[@]}" down

echo "[INFO] Building and starting Postgres, Redis, Backend (AI Model), and Frontend..."
"${COMPOSE_CMD[@]}" up --build -d

echo ""
echo "=========================================="
echo "    RETINA-Q System is Online via Docker  "
echo "    Frontend UI: http://localhost:3000    "
echo "    Backend API: http://localhost:8000    "
echo "=========================================="
echo ""
echo "To view live logs: ${COMPOSE_CMD[*]} logs -f"
echo "To shut down the system: ${COMPOSE_CMD[*]} down"
