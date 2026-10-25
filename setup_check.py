"""
Quick setup script – validates environment and creates required directories.
Run this after installation: python setup_check.py
"""
import sys
import os

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path

def check(name, fn, fix=None):
    try:
        fn()
        print(f"  ✅ {name}")
        return True
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        if fix:
            try:
                fix()
                print(f"     → Fixed!")
            except Exception as fe:
                print(f"     → Fix failed: {fe}")
        return False


print("\n🔍 RFP Intelligence Platform – Environment Check\n" + "=" * 50)

# 1. Python version
print("\n1. Python version")
check(
    f"Python {sys.version.split()[0]}",
    lambda: None if sys.version_info >= (3, 10) else (_ for _ in ()).throw(Exception("Python 3.10+ required"))
)

# 2. Required packages
print("\n2. Python packages")
packages = [
    ("pydantic", "pydantic"),
    ("PyMuPDF", "fitz"),
    ("pdfplumber", "pdfplumber"),
    ("BeautifulSoup4", "bs4"),
    ("trafilatura", "trafilatura"),
    ("pymongo", "pymongo"),
    ("rank_bm25", "rank_bm25"),
    ("sentence_transformers", "sentence_transformers"),
    ("fastapi", "fastapi"),
    ("streamlit", "streamlit"),
    ("langchain", "langchain"),
    ("langgraph", "langgraph"),
    ("loguru", "loguru"),
    ("httpx", "httpx"),
]

all_ok = True
for name, mod in packages:
    ok = check(name, lambda m=mod: __import__(m))
    if not ok:
        all_ok = False

# 3. Directories
print("\n3. Required directories")
dirs = ["data/bids", "data/bm25_indices", "output", "logs", "eval"]
for d in dirs:
    check(d, lambda p=d: Path(p).mkdir(parents=True, exist_ok=True) or None)

# 4. Ollama
print("\n4. Ollama connection")
try:
    import httpx
    resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
    tags = [m["name"] for m in resp.json().get("models", [])]
    check("Ollama running", lambda: None)
    check(
        f"nomic-embed-text model {'✓' if any('nomic' in t for t in tags) else '(run: ollama pull nomic-embed-text)'}",
        lambda: None if any("nomic" in t for t in tags) else (_ for _ in ()).throw(Exception("Not pulled"))
    )
    check(
        f"llama3.2 model {'✓' if any('llama3.2' in t or 'llama3' in t for t in tags) else '(run: ollama pull llama3.2)'}",
        lambda: None if any("llama3" in t for t in tags) else (_ for _ in ()).throw(Exception("Not pulled"))
    )
except Exception as e:
    print(f"  ❌ Ollama not reachable: {e}")
    print("     → Start Ollama: ollama serve")

# 5. .env file
print("\n5. Configuration")
check(
    ".env file exists",
    lambda: None if Path(".env").exists() else (_ for _ in ()).throw(Exception("Copy .env.example to .env and fill in MONGODB_URI")),
    lambda: Path(".env.example").read_text() and open(".env", "w").write(Path(".env.example").read_text())
)

# 6. MongoDB
print("\n6. MongoDB Atlas")
try:
    from dotenv import load_dotenv
    load_dotenv()
    uri = os.getenv("MONGODB_URI", "")
    if "mongodb+srv" in uri or "mongodb://" in uri:
        import pymongo
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        print("  ✅ MongoDB Atlas connected")
    else:
        print("  ⚠️  MONGODB_URI not set in .env (required for vector search)")
except Exception as e:
    print(f"  ❌ MongoDB connection failed: {e}")
    print("     → Check MONGODB_URI in .env file")

print("\n" + "=" * 50)
if all_ok:
    print("✅ All checks passed! You're ready to go.\n")
    print("Next steps:")
    print("  1. Add bid files to data/bids/Bid1/ and data/bids/Bid2/")
    print("  2. python main.py index --bid ./data/bids/Bid1 --bid-id Bid1")
    print("  3. python main.py extract --bid ./data/bids/Bid1 --bid-id Bid1")
    print("  4. streamlit run ui/app.py")
else:
    print("⚠️  Some checks failed. Fix the issues above and re-run.\n")
