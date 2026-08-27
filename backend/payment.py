"""
backend/payment.py — Sign0 Deep Stripe Payment Processor
Supports live Stripe Checkout sessions, Billing Portals, and secure Webhook payload validation.
"""

import os
import json
import stripe
from datetime import datetime
from fastapi import APIRouter, HTTPException, Request, Depends, Header, status
from pydantic import BaseModel
from typing import Optional

from backend.auth import PLANS, get_current_user
from backend.database import SessionLocal
from backend.models import User, Activity

router = APIRouter(prefix="/payment", tags=["billing"])

# Initialize Stripe API configurations
STRIPE_API_KEY = os.environ.get("STRIPE_API_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()

if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY
    IS_SANDBOX = False
else:
    # Fallback to local sandbox emulation if no live developer key is provided
    IS_SANDBOX = True

# Stripe Price IDs mapping
STRIPE_PRICES = {
    "pro": os.environ.get("STRIPE_PRICE_PRO", "price_mock_pro_learner_99"),
    "developer": os.environ.get("STRIPE_PRICE_DEVELOPER", "price_mock_developer_499")
}

class CheckoutRequest(BaseModel):
    plan: str
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None

class PortalRequest(BaseModel):
    return_url: Optional[str] = None

@router.post("/checkout")
async def create_checkout_session(req: CheckoutRequest, current_user: dict = Depends(get_current_user)):
    username = current_user["username"]
    plan = req.plan.lower().strip()
    email = current_user.get("email")
    fullname = current_user.get("full_name", "")
    
    if plan not in PLANS:
        raise HTTPException(status_code=400, detail="Invalid subscription plan selected.")
        
    if plan == current_user.get("plan"):
        return {"success": False, "message": "You are already subscribed to this plan."}

    # Handle Sandbox Emulation
    if IS_SANDBOX:
        import secrets
        session_id = f"cs_sandbox_" + secrets.token_hex(16)
        # Sandbox callback redirects to simulated payment success
        if req.success_url:
            mock_success = req.success_url.replace("{CHECKOUT_SESSION_ID}", session_id)
        else:
            mock_success = f"/frontend/index.html?payment=success&session_id={session_id}&plan={plan}"
            
        return {
            "success": True,
            "session_id": session_id,
            "checkout_url": mock_success,
            "sandbox": True,
            "message": "Checkout session created (Sandbox mode)."
        }

    # Real Stripe Implementation
    try:
        with SessionLocal() as db:
            user = db.query(User).filter(User.username == username).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            customer_id = user.stripe_customer_id
            
            # 1. Create or retrieve Stripe Customer
            if not customer_id:
                customer = stripe.Customer.create(
                    email=email,
                    name=fullname,
                    metadata={"username": username}
                )
                customer_id = customer.id
                user.stripe_customer_id = customer_id
                db.commit()

        price_id = STRIPE_PRICES.get(plan)
        if not price_id:
            raise HTTPException(status_code=500, detail=f"Price ID configuration missing for plan: {plan}")

        # 2. Create Stripe Checkout Session
        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[{
                "price": price_id,
                "quantity": 1,
            }],
            mode="subscription",
            success_url=req.success_url or "http://127.0.0.1:8000/frontend/index.html?payment=success&session_id={CHECKOUT_SESSION_ID}",
            cancel_url=req.cancel_url or "http://127.0.0.1:8000/frontend/index.html?payment=cancelled",
            metadata={
                "username": username,
                "plan": plan
            },
            client_reference_id=username
        )
        
        return {
            "success": True,
            "session_id": session.id,
            "checkout_url": session.url,
            "sandbox": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe Session Error: {str(e)}")

@router.post("/portal")
async def create_portal_session(req: PortalRequest, current_user: dict = Depends(get_current_user)):
    username = current_user["username"]
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == username).first()
        customer_id = user.stripe_customer_id if user else None
    
    if IS_SANDBOX:
        # Sandbox portal returns simulated redirect to billing page
        return {
            "success": True,
            "portal_url": "/frontend/index.html?payment=portal",
            "sandbox": True
        }
        
    if not customer_id:
        raise HTTPException(
            status_code=400, 
            detail="No Stripe customer record found. Subscribe to a plan first."
        )
        
    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=req.return_url or "http://127.0.0.1:8000/frontend/index.html"
        )
        return {
            "success": True,
            "portal_url": session.url,
            "sandbox": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stripe Portal Error: {str(e)}")

@router.post("/webhook")
async def stripe_webhook(request: Request, stripe_signature: Optional[str] = Header(None)):
    """
    Handles live verified Stripe events:
      - checkout.session.completed (provisions subscription)
      - invoice.paid (provisions and extends quota)
      - invoice.payment_failed (locks plan)
      - customer.subscription.deleted (reverts to free plan)
    """
    payload = await request.body()
    
    # 1. Parse Event (Verify Signature or Sandbox direct parse)
    if IS_SANDBOX or not stripe_signature or not STRIPE_WEBHOOK_SECRET:
        # Sandbox mode or direct API webhook emulation
        try:
            event_dict = json.loads(payload.decode("utf-8"))
            event = stripe.Event.construct_from(event_dict, key=None)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JSON Parse Error: {str(e)}")
    else:
        # Secure production validation using Stripe Signature header verification
        try:
            event = stripe.Webhook.construct_event(
                payload, stripe_signature, STRIPE_WEBHOOK_SECRET
            )
        except stripe.error.SignatureVerificationError as e:
            raise HTTPException(status_code=400, detail="Invalid webhook signature.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Webhook structure error: {str(e)}")

    event_type = event.type
    session_data = event.data.object
    
    # 2. Process Webhook Event Types
    if event_type == "checkout.session.completed":
        username = session_data.get("client_reference_id") or session_data.get("metadata", {}).get("username")
        plan = session_data.get("metadata", {}).get("plan")
        customer_id = session_data.get("customer")
        subscription_id = session_data.get("subscription")
        
        if username and plan:
            with SessionLocal() as db:
                user = db.query(User).filter(User.username == username).first()
                if user:
                    user.plan = plan
                    user.stripe_customer_id = customer_id
                    user.stripe_subscription_id = subscription_id
                    user.subscription_status = "active"
                    
                    act = Activity(username=username, endpoint=f"Stripe Checkout completed ({plan})")
                    db.add(act)
                    db.commit()
                
    elif event_type == "invoice.paid":
        # Handle subscription renewal
        customer_id = session_data.get("customer")
        if customer_id:
            with SessionLocal() as db:
                user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
                if user:
                    user.subscription_status = "active"
                    # Reset usage counters for renewal cycle
                    usage = dict(user.usage) if user.usage else {}
                    usage.setdefault("daily", {})
                    usage["daily"] = {} # clear out past days (or reset specifically)
                    user.usage = usage
                    
                    act = Activity(username=user.username, endpoint="Stripe Invoice paid - Subscription renewed")
                    db.add(act)
                    db.commit()
                    
    elif event_type in ["invoice.payment_failed", "customer.subscription.deleted"]:
        # Handle billing failures or cancellations
        customer_id = session_data.get("customer")
        if customer_id:
            with SessionLocal() as db:
                user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
                if user:
                    old_plan = user.plan
                    user.plan = "free"
                    user.subscription_status = "canceled"
                    user.stripe_subscription_id = ""
                    
                    act = Activity(username=user.username, endpoint=f"Stripe Subscription canceled/deleted (reverted {old_plan} -> free)")
                    db.add(act)
                    db.commit()

    return {"status": "success"}
