#!/usr/bin/env python
"""
Quick start guide - Run this to get started with URL Definer development
"""

import os
import subprocess
import sys

def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n📌 {description}...")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} complete!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error code {e.returncode}")
        return False

def main():
    print("=" * 60)
    print("URL Definer - Quick Start Guide")
    print("=" * 60)
    
    # Check if virtual environment is activated
    if sys.prefix == sys.base_prefix:
        print("\n⚠️  Virtual environment not activated!")
        print("Please activate your virtual environment first:")
        print("  Windows: venv\\Scripts\\activate")
        print("  macOS/Linux: source venv/bin/activate")
        return
    
    print("\n✓ Virtual environment activated")
    
    # Create .env from .env.example if it doesn't exist
    if not os.path.exists('.env'):
        print("\n📌 Creating .env from .env.example...")
        if os.path.exists('.env.example'):
            with open('.env.example', 'r') as src, open('.env', 'w') as dst:
                dst.write(src.read())
            print("✓ .env created - edit it to add your API keys")
        else:
            print("⚠️  .env.example not found")
    else:
        print("✓ .env already exists")
    
    # Install dependencies
    if not run_command("pip install -q -r requirements.txt", "Installing dependencies"):
        return
    
    # Check if model exists
    if not os.path.exists('url_classifier.pkl') or not os.path.exists('feature_extractor.pkl'):
        print("\n⚠️  Model not found. Training model...")
        if not run_command("python scripts/train.py", "Training ML model"):
            return
    else:
        print("\n✓ Model artifacts found")
    
    # Run tests (optional)
    print("\n📌 Run tests? (y/n): ", end='')
    if input().lower() == 'y':
        run_command("pytest tests/ -v", "Running tests")
    
    # Start app
    print("\n" + "=" * 60)
    print("✓ Setup complete! Starting app...")
    print("=" * 60)
    print("\nThe app will be available at: http://127.0.0.1:5000/")
    print("Press Ctrl+C to stop\n")
    
    os.system("python app.py")

if __name__ == "__main__":
    main()
