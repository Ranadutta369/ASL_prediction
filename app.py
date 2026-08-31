"""
app.py — Root Proxy for Sign0 FastAPI Application
Allows Render and other cloud platforms to start using either `uvicorn app:app` or `uvicorn backend.app:app`.
"""
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.app import app

__all__ = ["app"]
