"""
One-Command Launcher for RFP Intelligence Platform.

Usage:
    python run.py
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    print("=" * 60)
    print("🚀  RFP Intelligence Platform — One-Command Startup")
    print("=" * 60)

    # 1. Check Python version
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required. Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

    # 2. Setup .env if missing
    env_file = ROOT / ".env"
    env_example = ROOT / ".env.example"
    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print("📋 Created .env from .env.example (please update credentials if needed)")
        else:
            print("⚠️  Warning: .env and .env.example not found.")
    else:
        print("✅ Environment file (.env) found.")

    # 3. Check / Install dependencies
    try:
        import streamlit
        import pymupdf
        import loguru
        print("✅ Core dependencies verified.")
    except ImportError:
        print("📦 Installing required dependencies from requirements.txt...")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")]
        subprocess.check_call(cmd)
        print("✅ Dependencies installed.")

    # 4. Check data directories
    bm25_dir = ROOT / "data" / "bm25_indices"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 5. Launch Streamlit UI
    ui_script = ROOT / "ui" / "app.py"
    port = os.getenv("STREAMLIT_PORT", "8501")

    print("-" * 60)
    print(f"🌐 Starting Streamlit Web UI on http://localhost:{port}")
    print("💡 Press Ctrl+C in this terminal to stop the server.")
    print("-" * 60)

    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_script),
        "--server.port",
        port,
        "--browser.gatherUsageStats",
        "false",
    ]

    try:
        subprocess.run(streamlit_cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        print("\n👋 Platform stopped.")

if __name__ == "__main__":
    main()
