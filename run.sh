#!/bin/bash
# ═══════════════════════════════════════════════════════
# AI-Based Manufacturing Efficiency Classification
# Quick Start Script
# ═══════════════════════════════════════════════════════

echo "🏭 Manufacturing Efficiency AI — Setup & Launch"
echo "================================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.10+"
    exit 1
fi

echo "✅ Python3 found: $(python3 --version)"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt --quiet 2>/dev/null || pip install --break-system-packages -r requirements.txt --quiet

# Check if models exist
if [ ! -f "models/best_model.pkl" ]; then
    echo ""
    echo "🧠 Training models (first run)..."
    python3 analysis.py
else
    echo "✅ Pre-trained models found"
fi

# Optional: Set up Gemini API
if [ -f ".env" ]; then
    echo "✅ Gemini API key found (.env)"
else
    echo "⚠️  No .env file — AI Insights tab will be disabled"
    echo "   To enable, create .env with: GEMINI_API_KEY=your_key"
fi

# Launch dashboard
echo ""
echo "🚀 Launching dashboard..."
echo "   Open: http://localhost:8502"
echo "   Press Ctrl+C to stop"
echo ""
streamlit run app.py --server.port 8502 --server.headless true
