"""
download_data.py
ai_10022200118 — Obour Adomako Tawiah — CS4241 (2026)

Downloads the two source datasets.  Run this once on a machine with
internet access.  Streamlit Cloud runs it automatically on first boot
(see streamlit_app.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import RAW_DIR  # noqa: E402

URLS = {
    "Ghana_Election_Result.csv": (
        "https://raw.githubusercontent.com/GodwinDansoAcity/acitydataset/"
        "main/Ghana_Election_Result.csv"
    ),
    "2025-Budget-Statement.pdf": (
        "https://mofep.gov.gh/sites/default/files/budget-statements/"
        "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    ),
}


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in URLS.items():
        dest = RAW_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {filename} already present ({dest.stat().st_size} bytes)")
            continue
        print(f"[get ] {url}")
        try:
            r = requests.get(url, timeout=60,
                             headers={"User-Agent": "ai-acity-rag/1.0"})
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"[save] {dest} ({len(r.content)} bytes)")
        except Exception as e:
            print(f"[err ] could not download {filename}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
