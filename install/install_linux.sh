#!/bin/bash
cd "$(dirname "$0")/.."

echo "=========================================="
echo "Installing Plott3r dependencies for Linux"
echo "=========================================="

sudo apt update
sudo apt install -y python3 python3-pip python3-venv git libgl1 libglib2.0-0

# Setup virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "=========================================="
echo "Installation complete!"
echo "To start the server, run:"
echo "bash start_linux.sh"
echo "=========================================="
