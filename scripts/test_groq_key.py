"""Tiny diagnostic — loads .env, prints the key prefix, hits Groq once."""
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
env_path = ROOT / ".env"
print(f".env path: {env_path}")
print(f".env exists: {env_path.exists()}")
print(f".env size: {env_path.stat().st_size if env_path.exists() else 'N/A'} bytes")
load_dotenv(env_path)

key = os.environ.get("GROQ_API_KEY", "")
print(f"Key loaded: {bool(key)}")
print(f"Key length: {len(key)}")
print(f"Key prefix: {key[:8] if key else '(empty)'}")
print(f"Key suffix: {key[-4:] if key else '(empty)'}")
print(f"Starts with gsk_: {key.startswith('gsk_') if key else False}")
print(f"Contains spaces: {' ' in key}")
print(f"Contains quotes: {chr(34) in key or chr(39) in key}")

if not key:
    print("ABORTING: no key loaded.")
    raise SystemExit(1)

print("\n--- calling Groq ---")
from groq import Groq
client = Groq(api_key=key)
try:
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "say hi"}],
        max_tokens=5,
    )
    print("SUCCESS:", resp.choices[0].message.content)
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:200])
