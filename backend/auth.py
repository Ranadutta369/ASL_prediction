"""
backend/auth.py — Sign0 AI Engine Authentication & Plan Authorization
JWT-based authentication with bcrypt hashing, rate limiting, and subscription plan definitions.
"""

import os
import time
import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from jose import JWTError, jwt

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

# Read stable JWT secret from environment or local file
SECRET_KEY = os.environ.get("SIGN0_SECRET_KEY", "").strip()
if not SECRET_KEY:
    _SECRET_FILE = Path(__file__).resolve().parent / "data" / ".jwt_secret"
    _SECRET_FILE.parent.mkdir(exist_ok=True, parents=True)
    if _SECRET_FILE.exists():
        SECRET_KEY = _SECRET_FILE.read_text().strip()
    else:
        SECRET_KEY = secrets.token_hex(32)
        try:
            _SECRET_FILE.write_text(SECRET_KEY)
        except Exception:
            pass  # read-only disk fallback

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days (remember-me session)

# ──────────────────────────────────────────────
# Subscription Plans
# ──────────────────────────────────────────────

PLANS: Dict[str, dict] = {
    "free": {
        "name": "Free",
        "price": 0,
        "price_label": "$0/month",
        "daily_quota": 20,
        "monthly_quota": 200,
        "features": {
            "static_mlp": True,
            "sequence_transformer": False,
            "diffusion_synthesizer": False,
            "dict_visualizer": False,
            "tts": True,
            "custom_recorder": False,
        },
        "description": "Basic sign recognition for learning & testing.",
        "highlights": [
            "20 static predictions per day",
            "Real-time MediaPipe visual feedback",
            "Text-to-Speech audio reader",
            "Basic alphabetical dictionary cards",
        ]
    },
    "pro": {
        "name": "Pro Learner",
        "price": 9,
        "price_label": "$9/month",
        "daily_quota": 500,
        "monthly_quota": 5000,
        "features": {
            "static_mlp": True,
            "sequence_transformer": True,
            "diffusion_synthesizer": False,
            "dict_visualizer": True,
            "tts": True,
            "custom_recorder": False,
        },
        "description": "Ideal for students and speech-impaired individuals.",
        "highlights": [
            "500 predictions per day",
            "Spatial-Temporal Transformer (seq) access",
            "Interactive 3D sign visual dictionary player",
            "WebRTC camera feed calibration",
            "Gamified practice challenge modes",
        ]
    },
    "developer": {
        "name": "Developer / Enterprise",
        "price": 49,
        "price_label": "$49/month",
        "daily_quota": 999999,  # Unlimited
        "monthly_quota": 999999,
        "features": {
            "static_mlp": True,
            "sequence_transformer": True,
            "diffusion_synthesizer": True,
            "dict_visualizer": True,
            "tts": True,
            "custom_recorder": True,
        },
        "description": "Unlimited API access for developers and research labs.",
        "highlights": [
            "Unlimited requests (no daily quotas)",
            "Text-to-Sign DDPM Diffusion Synthesizer access",
            "Interactive custom gesture recording",
            "Private API Key provisioning",
            "Commercial integration rights",
        ]
    }
}

# ──────────────────────────────────────────────
# Passwords & JWT Cryptography
# ──────────────────────────────────────────────

import bcrypt

def hash_password(password: str) -> str:
    pwd_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        pwd_bytes = plain_password.encode('utf-8')[:72]
        return bcrypt.checkpw(pwd_bytes, hashed_password.encode('utf-8'))
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid token. Please log in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# User Database Helpers (SQLAlchemy)
# ──────────────────────────────────────────────

from backend.database import SessionLocal, engine
from backend.models import User, Activity, Base
from fastapi import Request

# Create tables
Base.metadata.create_all(bind=engine)

def seed_default_user():
    with SessionLocal() as db:
        if not db.query(User).filter(User.username == "dev_demo").first():
            dev_user = User(
                username="dev_demo",
                email="demo@sign0.ai",
                full_name="Demo Developer",
                hashed_password=hash_password("asldemo2026"),
                role="developer",
                plan="developer",
                api_key="sign0_live_dev_k8s_9281aef10",
                usage={}
            )
            db.add(dev_user)
            db.commit()

seed_default_user()

# ──────────────────────────────────────────────
# Rate Limiting & Gating
# ──────────────────────────────────────────────

def get_current_date_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")

def get_current_month_str() -> str:
    return datetime.utcnow().strftime("%Y-%m")

def record_api_call(username: str, endpoint: str):
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return
            
        day = get_current_date_str()
        month = get_current_month_str()
        
        # SQLAlchemy JSON columns need reassignment to trigger updates
        usage = dict(user.usage) if user.usage else {}
        usage.setdefault("daily", {})
        usage.setdefault("monthly", {})
        
        usage["daily"][day] = usage["daily"].get(day, 0) + 1
        usage["monthly"][month] = usage["monthly"].get(month, 0) + 1
        
        user.usage = usage
        
        act = Activity(username=username, endpoint=endpoint)
        db.add(act)
        db.commit()

def get_usage(username: str) -> dict:
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == username).first()
        plan = user.plan if user else "free"
        limits = PLANS.get(plan, PLANS["free"])
        
        usage = user.usage if user and user.usage else {}
        day = get_current_date_str()
        month = get_current_month_str()
        
        queries_today = usage.get("daily", {}).get(day, 0)
        queries_month = usage.get("monthly", {}).get(month, 0)
        
        daily_quota = limits["daily_quota"]
        monthly_quota = limits["monthly_quota"]
        
        return {
            "plan": plan,
            "queries_today": queries_today,
            "queries_month": queries_month,
            "daily_quota": daily_quota,
            "monthly_quota": monthly_quota,
            "daily_remaining": max(0, daily_quota - queries_today),
            "monthly_remaining": max(0, monthly_quota - queries_month),
        }

def check_rate_limit(username: str) -> int:
    """Returns the remaining requests. Raises HTTP 429 if quota exceeded."""
    usage = get_usage(username)
    if usage["queries_today"] >= usage["daily_quota"]:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily request quota exceeded. Please upgrade your plan."
        )
    return usage["daily_remaining"]

# ──────────────────────────────────────────────
# FastAPI Dependencies
# ──────────────────────────────────────────────

async def get_current_user(request: Request) -> dict:
    token = request.cookies.get("access_token")
    if not token:
        # Fallback to header for developers or old clients
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is missing. Please sign in.",
        )
        
    payload = decode_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload.")
    
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account not found.")
        
        user_dict = {
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "plan": user.plan,
            "api_key": user.api_key,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "usage": get_usage(user.username)
        }
    return user_dict

async def get_optional_user(request: Request) -> Optional[dict]:
    try:
        return await get_current_user(request)
    except Exception:
        return None
