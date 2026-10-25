#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  RFP Intelligence Platform - Quick Start (Linux/macOS)"
echo "============================================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH."
    exit 1
fi

python3 run.py
