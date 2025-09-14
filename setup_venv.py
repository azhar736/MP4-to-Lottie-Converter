#!/usr/bin/env python3
"""
MP4 to Lottie Converter - Easy Setup Script

Creates virtual environment and installs all dependencies automatically.
Solves "externally-managed-environment" errors on macOS/Linux systems.
"""

import subprocess
import sys
import os
import venv
from pathlib import Path


def create_virtual_environment():
    """Create a virtual environment in the project directory."""
    venv_path = Path("venv")

    if venv_path.exists():
        print("✓ Virtual environment already exists")
        return True

    try:
        print("Creating virtual environment...")
        venv.create(venv_path, with_pip=True)
        print("✓ Virtual environment created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False


def get_venv_python():
    """Get the path to the Python executable in the virtual environment."""
    if sys.platform == "win32":
        return Path("venv/Scripts/python.exe")
    else:
        return Path("venv/bin/python")


def get_venv_pip():
    """Get the path to pip in the virtual environment."""
    if sys.platform == "win32":
        return Path("venv/Scripts/pip")
    else:
        return Path("venv/bin/pip")


def install_dependencies():
    """Install dependencies in the virtual environment."""
    pip_path = get_venv_pip()

    if not pip_path.exists():
        print("✗ Virtual environment pip not found")
        return False

    try:
        print("Installing dependencies in virtual environment...")
        subprocess.check_call([str(pip_path), "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False


def test_installation():
    """Test the installation in the virtual environment."""
    python_path = get_venv_python()

    if not python_path.exists():
        print("✗ Virtual environment Python not found")
        return False

    try:
        print("Testing installation...")

        # Test importing required modules
        test_script = """
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import cv2
    import PIL
    import numpy
    import pathvalidate
    from src.utils import ValidationUtils
    from src.lottie_converter import LottieConverter
    
    # Basic functionality test
    assert ValidationUtils.validate_dimensions(1920, 1080) == True
    converter = LottieConverter(1920, 1080, 30.0)
    assert converter.width == 1920
    
    print("✓ All tests passed!")
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)
"""

        # Write test script to temporary file
        with open("temp_test.py", "w") as f:
            f.write(test_script)

        # Run test in virtual environment
        subprocess.check_call([str(python_path), "temp_test.py"])

        # Clean up
        os.remove("temp_test.py")

        print("✓ Installation test passed")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Installation test failed: {e}")
        if os.path.exists("temp_test.py"):
            os.remove("temp_test.py")
        return False


def create_activation_scripts():
    """Create convenient activation scripts."""

    # Unix/macOS activation script
    activate_script = """#!/bin/bash
# MP4 to Lottie Converter - Virtual Environment Activation

echo "Activating MP4 to Lottie Converter virtual environment..."

# Activate virtual environment
source venv/bin/activate

# Show Python version and location
echo "Python: $(python --version)"
echo "Location: $(which python)"

echo ""
echo "Virtual environment activated!"
echo "You can now run:"
echo "  python main.py                    # Launch GUI"
echo "  python main.py video.mp4          # Convert video"
echo "  python main.py --help             # Show help"
echo ""
echo "To deactivate, run: deactivate"
"""

    with open("activate.sh", "w") as f:
        f.write(activate_script)

    # Make executable
    os.chmod("activate.sh", 0o755)

    # Windows activation script
    activate_bat = """@echo off
REM MP4 to Lottie Converter - Virtual Environment Activation

echo Activating MP4 to Lottie Converter virtual environment...

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Show Python version and location
python --version
where python

echo.
echo Virtual environment activated!
echo You can now run:
echo   python main.py                    # Launch GUI
echo   python main.py video.mp4          # Convert video
echo   python main.py --help             # Show help
echo.
echo To deactivate, run: deactivate
"""

    with open("activate.bat", "w") as f:
        f.write(activate_bat)

    print("✓ Created activation scripts:")
    print("  - activate.sh (Unix/macOS)")
    print("  - activate.bat (Windows)")


def create_run_scripts():
    """Create convenient run scripts that use the virtual environment."""

    # Unix/macOS run script
    run_script = """#!/bin/bash
# MP4 to Lottie Converter - Direct Run Script

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup_venv.py first."
    exit 1
fi

# Use virtual environment Python
VENV_PYTHON="venv/bin/python"

# Run the application with all arguments
$VENV_PYTHON main.py "$@"
"""

    with open("run_venv.sh", "w") as f:
        f.write(run_script)

    # Make executable
    os.chmod("run_venv.sh", 0o755)

    # Windows run script
    run_bat = """@echo off
REM MP4 to Lottie Converter - Direct Run Script

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Please run setup_venv.py first.
    exit /b 1
)

REM Use virtual environment Python
set VENV_PYTHON=venv\\Scripts\\python.exe

REM Run the application with all arguments
%VENV_PYTHON% main.py %*
"""

    with open("run_venv.bat", "w") as f:
        f.write(run_bat)

    print("✓ Created direct run scripts:")
    print("  - run_venv.sh (Unix/macOS)")
    print("  - run_venv.bat (Windows)")


def main():
    """Main setup function."""
    print("MP4 to Lottie Converter - Virtual Environment Setup")
    print("=" * 60)
    print()

    # Step 1: Create virtual environment
    if not create_virtual_environment():
        print("\nSetup failed at virtual environment creation.")
        return False

    # Step 2: Install dependencies
    if not install_dependencies():
        print("\nSetup failed at dependency installation.")
        return False

    # Step 3: Test installation
    if not test_installation():
        print("\nSetup failed at installation testing.")
        return False

    # Step 4: Create convenience scripts
    create_activation_scripts()
    create_run_scripts()

    # Success message
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("Your MP4 to Lottie Converter is ready to use!")
    print()
    print("Quick Start Options:")
    print()
    print("Option 1 - Direct Run (Recommended):")
    if sys.platform == "win32":
        print("  run_venv.bat                    # Launch GUI")
        print("  run_venv.bat video.mp4          # Convert video")
        print("  run_venv.bat --help             # Show help")
    else:
        print("  ./run_venv.sh                   # Launch GUI")
        print("  ./run_venv.sh video.mp4         # Convert video")
        print("  ./run_venv.sh --help            # Show help")
    print()
    print("Option 2 - Manual Activation:")
    if sys.platform == "win32":
        print("  activate.bat                    # Activate environment")
    else:
        print("  source activate.sh              # Activate environment")
    print("  python main.py                  # Then run normally")
    print()
    print("Option 3 - Command Line:")
    if sys.platform == "win32":
        print("  venv\\Scripts\\python.exe main.py")
    else:
        print("  venv/bin/python main.py")
    print()
    print("The virtual environment isolates all dependencies,")
    print("solving the 'externally-managed-environment' error.")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during setup: {e}")
        sys.exit(1)
