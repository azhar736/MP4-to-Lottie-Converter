@echo off
REM MP4 to Lottie Converter - Direct Run Script

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Please run setup_venv.py first.
    exit /b 1
)

REM Use virtual environment Python
set VENV_PYTHON=venv\Scripts\python.exe

REM Run the application with all arguments
%VENV_PYTHON% main.py %*
