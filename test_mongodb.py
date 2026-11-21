import os

import certifi
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
uri = os.getenv("MONGODB_URL_KEY")
if not uri:
    raise RuntimeError("MONGODB_URL_KEY is missing. Add it to your .env file.")

client = MongoClient(
    uri,
    tlsCAFile=certifi.where(),
    serverSelectionTimeoutMS=10000,
)

try:
    client.admin.command("ping")
    print("Successfully connected to MongoDB Atlas.")
except Exception as exc:
    print(f"MongoDB connection failed: {exc}")
finally:
    client.close()
