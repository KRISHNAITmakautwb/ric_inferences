#!/bin/bash
# restart_ric.sh — Safe restart for RIC server
# Usage: bash restart_ric.sh

cd "$(dirname "$0")"

echo "🛑 Stopping any existing RIC server on port 8000..."
fuser -k 8000/tcp 2>/dev/null
pkill -f "python server.py" 2>/dev/null
sleep 1

echo "✅ Port 8000 is free. Starting RIC server..."
source venv/bin/activate
python server.py
