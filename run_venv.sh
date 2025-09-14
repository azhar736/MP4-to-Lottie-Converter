#!/bin/bash
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
