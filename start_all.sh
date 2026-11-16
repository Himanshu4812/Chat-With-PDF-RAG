#!/bin/bash

echo "=================================="
echo "Starting Ollama PDF RAG Services"
echo "=================================="
echo ""

# Start FastAPI backend (uses isolated .venv if present)
echo "🚀 Starting FastAPI backend (port 8001)..."
if [ -f ".venv/Scripts/python.exe" ]; then PYTHON=".venv/Scripts/python.exe"; else PYTHON="python3"; fi
$PYTHON run_api.py &
FASTAPI_PID=$!
echo "   FastAPI PID: $FASTAPI_PID"
echo ""

# Wait for FastAPI to start
sleep 5

# Start Next.js frontend
echo "🎨 Starting Next.js frontend (port 3000)..."
cd web-ui && pnpm dev &
NEXTJS_PID=$!
echo "   Next.js PID: $NEXTJS_PID"
cd ..
echo ""

echo "=================================="
echo "All services started!"
echo "=================================="
echo ""
echo "📡 Service URLs:"
echo "   • FastAPI Backend:  http://localhost:8001"
echo "   • Next.js Frontend: http://localhost:3000"
echo ""
echo "📚 API Documentation:"
echo "   • FastAPI Docs:     http://localhost:8001/docs"
echo "   • Health Check:     http://localhost:8001/api/v1/health"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for user interrupt
wait
