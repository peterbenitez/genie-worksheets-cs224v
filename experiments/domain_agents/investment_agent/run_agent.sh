#!/bin/bash

# Set up environment
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"
set -a
source ../../../.env
set +a

# Run the investment agent
../../../.venv/bin/python investment_agent.py
