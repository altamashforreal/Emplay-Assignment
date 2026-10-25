@echo off
echo ============================================================
echo   RFP Intelligence Platform - Quick Start (Windows)
echo ============================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    pause
    exit /b 1
)

python run.py
pause
