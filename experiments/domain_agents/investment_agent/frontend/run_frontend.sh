#!/bin/bash

# Run the Investment Agent Web Interface
# This will start a web server that you can access in your browser

echo "Starting Investment Agent Web Interface..."
echo ""
echo "Prerequisites:"
echo "  - PostgreSQL should be running"
echo "  - SUQL embedding server should be running (port 8500)"
echo ""

# Set up environment
# Add PostgreSQL to PATH based on OS (macOS with Homebrew)
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS with Homebrew
    export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
fi
set -a
source ../../../../.env
set +a

# Run Chainlit with the investment agent
../../../../.venv/bin/chainlit run app_investment_agent.py --port 8800

echo ""
echo "To access the web interface, open your browser to: http://localhost:8800"
