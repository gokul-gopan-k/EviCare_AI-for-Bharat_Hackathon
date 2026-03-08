#!/bin/bash
set -e

# EviCare Docker Entrypoint Script
# Supports running backend, frontend, or both services

SERVICE=${1:-both}

echo "Starting EviCare service: $SERVICE"

case $SERVICE in
  backend)
    echo "Starting FastAPI backend on port 8000..."
    exec uvicorn backend.main:app --host 0.0.0.0 --port 8000
    ;;
  
  frontend)
    echo "Starting Streamlit frontend on port 8501..."
    exec streamlit run frontend/main.py --server.port=8501 --server.address=0.0.0.0
    ;;
  
  both)
    echo "Starting both FastAPI backend and Streamlit frontend..."
    # Start backend in background
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    echo "Backend started with PID: $BACKEND_PID"
    
    # Wait a moment for backend to initialize
    sleep 3
    
    # Start frontend in foreground (keeps container running)
    echo "Starting frontend..."
    exec streamlit run frontend/main.py --server.port=8501 --server.address=0.0.0.0
    ;;
  
  *)
    echo "Error: Invalid service specified"
    echo "Usage: $0 {backend|frontend|both}"
    echo ""
    echo "Examples:"
    echo "  $0 backend   - Run only FastAPI backend"
    echo "  $0 frontend  - Run only Streamlit frontend"
    echo "  $0 both      - Run both services (default)"
    exit 1
    ;;
esac
