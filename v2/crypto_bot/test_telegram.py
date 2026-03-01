
import os
import requests
import sys

# Read from .env manually to debug formatting issues
env_path = "/opt/crypto_bot/crypto_bot/.env"
print(f"Reading {env_path}...")

token = None
chat_id = None

try:
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if key == "TELEGRAM_BOT_TOKEN":
                    token = val
                elif key == "TELEGRAM_CHAT_ID":
                    chat_id = val
except Exception as e:
    print(f"Error reading .env: {e}")
    sys.exit(1)

print(f"Token found: {'YES' if token else 'NO'}")
print(f"Chat ID found: {'YES' if chat_id else 'NO'}")

if not token or not chat_id:
    print("❌ Missing credentials in .env")
    sys.exit(1)

print(f"Attempting to send message to {chat_id}...")
url = f"https://api.telegram.org/bot{token}/sendMessage"
payload = {"chat_id": chat_id, "text": "🔔 Test message from VPS"}

try:
    resp = requests.post(url, json=payload, timeout=10)
    print(f"Status Code: {resp.status_code}")
    print(f"Response: {resp.text}")
    if resp.status_code == 200:
        print("✅ Success!")
    else:
        print("❌ Failed.")
except Exception as e:
    print(f"❌ Connection error: {e}")
