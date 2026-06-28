#!/bin/bash
# Move to the project root directory where the script is located
cd "$(dirname "$0")"

echo "Starting Plott3r PC Control (macOS)..."
source venv/bin/activate
python computer_side/main.py
