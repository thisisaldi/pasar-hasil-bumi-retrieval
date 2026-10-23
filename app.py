"""Vercel entrypoint: re-export the FastAPI app from src/ (no install step needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from twotower.serve import app  # noqa: E402,F401
