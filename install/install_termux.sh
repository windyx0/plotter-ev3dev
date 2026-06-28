#!/bin/bash
cd "$(dirname "$0")/.."

echo "=========================================="
echo "Installing Plott3r dependencies for Termux"
echo "=========================================="

# Update and install basic dependencies
echo "Updating Termux packages..."
pkg update -y
pkg upgrade -y

# Install TUR repo (Termux User Repository) which has precompiled scientific packages
echo "Installing TUR repository..."
pkg install -y tur-repo x11-repo
pkg update -y

# Install Python and heavy precompiled dependencies via apt/pkg to avoid compiling errors
echo "Installing Python, NumPy, OpenCV, and SciKit-Image natively..."
pkg install python dbus dos2unix python-numpy python-pillow python-scipy matplotlib python-scikit-image opencv-python clang make ninja cmake pkg-config freetype libjpeg-turbo libpng libxml2 libxslt -y
# Install remaining pure-python dependencies via pip
echo "Installing remaining Python packages via pip..."
pip install --upgrade pip
pip install -r requirements.txt

echo "=========================================="
echo "Installation complete!"
echo "To start the server, run:"
echo "bash start_termux.sh"
echo "=========================================="
