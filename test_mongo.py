import pymongo
import os
from dotenv import load_dotenv

load_dotenv()
uri = os.getenv("MONGODB_URI", "")
print(f"Connecting to: {uri[:60]}...")
try:
    client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=8000)
    info = client.server_info()
    print("SUCCESS: MongoDB Atlas connected!")
    print(f"Version: {info['version']}")
    client.close()
except Exception as e:
    print(f"FAILED: {e}")
