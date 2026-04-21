"""Test Groq with the key passed as a command-line argument (no .env)."""
import sys
from groq import Groq

if len(sys.argv) < 2:
    print("Usage: py -3.13 scripts/test_groq_direct.py <your_full_key>")
    raise SystemExit(1)

key = sys.argv[1].strip()
print(f"Key length: {len(key)}, prefix: {key[:8]}, suffix: {key[-4:]}")

# check for any non-ASCII or invisible chars
bad = [(i, ch, hex(ord(ch))) for i, ch in enumerate(key) if not ch.isalnum() and ch != "_"]
if bad:
    print("SUSPICIOUS CHARS:", bad)
else:
    print("All chars are alphanumeric or underscore (clean).")

client = Groq(api_key=key)
try:
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "say hi in 3 words"}],
        max_tokens=10,
    )
    print("SUCCESS:", resp.choices[0].message.content)
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:250])
