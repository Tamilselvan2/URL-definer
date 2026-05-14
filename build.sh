#!/bin/bash
# Build script for render.com deployment
# This script will be run during the build phase on render.com

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Training the model..."
python scripts/train.py

echo "Build completed successfully!"
