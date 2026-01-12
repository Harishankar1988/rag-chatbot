#!/bin/bash

# PDF Chatbot RAG System Startup Script (LLaMA via Ollama)

# Check for development mode
if [ "$1" = "--dev" ]; then
    echo "🔧 Development mode - Only starting backend..."
    source venv/bin/activate
    cd backend
    python app.py
    exit 0
fi

echo "🚀 Starting PDF Chatbot RAG System (LLaMA + Ollama)..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3.10 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "Python version: $(python --version)"

# Ensure we're using venv Python and pip
echo "📦 Using Python: $(which python)"
echo "📦 Using pip: $(which pip)"

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Verify Flask installation
echo "🔍 Verifying Flask installation..."
if ! python -c "import flask; print('Flask installed successfully')" 2>/dev/null; then
    echo "❌ Flask installation failed!"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration (LLM_MODEL, OpenSearch settings)"
fi

# Check if Ollama is running
echo "🦙 Checking Ollama service..."
if ! curl -s http://localhost:11434 >/dev/null 2>&1; then
    echo "❌ Ollama is not running."
    echo "➡️  Start it using: ollama serve"
    exit 1
fi
echo "✅ Ollama is running"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "🐳 Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Start OpenSearch if not running
if ! docker ps | grep -q "pdf-chatbot-opensearch"; then
    echo "🐳 Starting OpenSearch..."
    docker-compose up -d

    echo "⏳ Waiting for OpenSearch to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:9200 >/dev/null 2>&1; then
            echo "✅ OpenSearch is ready!"
            break
        fi
        echo "   Attempt $i/30..."
        sleep 2
    done
fi

# Kill any existing backend on port 5001
echo "🧹 Checking for existing backend on port 5001..."
OLD_PID=$(lsof -ti :5001)
if [ ! -z "$OLD_PID" ]; then
    echo "🛑 Killing existing backend process: $OLD_PID"
    kill -9 $OLD_PID
    sleep 1
fi

# Start backend server in background
echo "🔧 Starting backend server..."
cd backend
source ../venv/bin/activate
python app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to be ready
echo "⏳ Waiting for backend to be ready..."
for i in {1..15}; do
    if curl -s http://localhost:5001/health >/dev/null 2>&1; then
        echo "✅ Backend is ready!"
        break
    fi
    echo "   Attempt $i/15..."
    sleep 1
done

# Start frontend
echo "🖥️  Starting frontend..."
cd frontend
source ../venv/bin/activate
echo "🌐 Frontend will open at: http://localhost:8501"
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# Cleanup function
cleanup() {
    echo "🛑 Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Set up trap for cleanup
trap cleanup SIGINT SIGTERM

# Wait for frontend
wait