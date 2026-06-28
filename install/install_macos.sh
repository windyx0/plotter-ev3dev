#!/bin/bash
cd "$(dirname "$0")/.."

echo "=========================================="
echo "Installing Plott3r dependencies for macOS"
echo "=========================================="

# Check if brew is installed
if ! command -v brew &> /dev/null
then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo "Updating Homebrew and installing Python 3..."
brew update
brew install python3

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
echo "bash start_macos.sh"
echo "=========================================="
