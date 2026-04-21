"""Diagnostic — loads .env, tries OpenRouter free models in sequence
until one accepts the request, reports which one worked.

Primary: openrouter/free (auto-router that picks any healthy free model).
"""
from pathlib import Path
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
env_path = ROOT / ".env"
print(f".env path: {env_path}")
print(f".env exists: {env_path.exists()}")
load_dotenv(env_path)

key = (os.environ.get("LLM_API_KEY")
       or os.environ.get("OPENROUTER_API_KEY", ""))
print(f"Key loaded: {bool(key)}")
print(f"Key length: {len(key)}")
print(f"Key prefix: {key[:10] if key else '(empty)'}")

if not key:
    print("ABORTING: no key loaded.")
    raise SystemExit(1)

from openai import OpenAI
client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")

MODELS = [
    "openrouter/free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

print("\n--- trying OpenRouter free models in order ---\n")
for m in MODELS:
    print(f"Trying {m} ...", end=" ", flush=True)
    try:
        resp = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": "say hi in 3 words"}],
            max_tokens=20,
            extra_headers={
                "HTTP-Referer": "https://github.com/ai_10022200118",
                "X-Title": "ai_10022200118 test",
            },
        )
        content = (resp.choices[0].message.content or "") if resp.choices else ""
        if content.strip():
            print(f"SUCCESS ({content.strip()[:60]})")
            print(f"\n>>> USE THIS MODEL in .env:  LLM_MODEL={m}")
            raise SystemExit(0)
        else:
            print("empty response")
    except Exception as e:
        msg = str(e)[:120]
        print(f"FAIL ({type(e).__name__}: {msg})")

print("\nAll free models are currently rate-limited or unavailable.")
print("Wait 2-3 minutes and retry.")
