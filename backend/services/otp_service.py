"""
OTP Service
Handles OTP generation, storage, and verification
Uses Twilio Verify API for production SMS
"""

import os
import random
import string
from datetime import datetime, timedelta
from typing import Optional, Dict
from dotenv import load_dotenv

load_dotenv()

# In-memory OTP storage (for dev mode only, Twilio handles this in production)
_otp_store: Dict[str, Dict] = {}

# Configuration
OTP_LENGTH = int(os.getenv("OTP_LENGTH", "6"))
OTP_EXPIRY_MINUTES = int(os.getenv("OTP_EXPIRY_MINUTES", "5"))
OTP_MAX_ATTEMPTS = int(os.getenv("OTP_MAX_ATTEMPTS", "3"))

# For development: print OTP to console instead of sending real SMS
DEV_MODE = os.getenv("DEBUG", "true").lower() == "true"

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_VERIFY_SERVICE_SID = os.getenv("TWILIO_VERIFY_SERVICE_SID")
OTP_BYPASS_FOR_TESTING = os.getenv("OTP_BYPASS_FOR_TESTING", "false").strip().lower() in {"1", "true", "yes", "on"}

# Initialize Twilio client (lazy loading)
_twilio_client = None


def get_twilio_client():
    """Get or create Twilio client (lazy initialization)"""
    global _twilio_client
    if _twilio_client is None and not DEV_MODE:
        try:
            from twilio.rest import Client
            _twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            print("✅ Twilio client initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize Twilio: {e}")
    return _twilio_client


class OTPService:
    """
    OTP Service for mobile verification
    
    - DEV_MODE (DEBUG=true): Prints OTP to console, no SMS sent
    - PRODUCTION (DEBUG=false): Uses Twilio Verify API for real SMS
    
    Twilio Verify handles OTP generation, storage, expiry, and rate limiting.
    """
    
    @staticmethod
    def generate_otp(length: int = OTP_LENGTH) -> str:
        """Generate a numeric OTP (used in dev mode only)"""
        return ''.join(random.choices(string.digits, k=length))
    
    @staticmethod
    def _format_phone(phone: str) -> str:
        """Format phone number to E.164 format for India"""
        phone = phone.strip().replace(" ", "").replace("-", "")
        # Add +91 if not present (India)
        if not phone.startswith("+"):
            if phone.startswith("91") and len(phone) == 12:
                phone = "+" + phone
            elif len(phone) == 10:
                phone = "+91" + phone
        return phone
    
    @staticmethod
    def send_otp(phone: str, purpose: str = "registration") -> Dict:
        """
        Send OTP to phone number
        
        - DEV_MODE: Generates OTP locally, prints to console
        - PRODUCTION: Uses Twilio Verify to send real SMS
        
        Args:
            phone: Phone number (with or without country code)
            purpose: Purpose of OTP (registration, login, reset)
            
        Returns:
            Dictionary with success status and message
        """
        phone = OTPService._format_phone(phone)

        # ===== TEST BYPASS MODE: Skip SMS provider and allow immediate OTP flow =====
        if OTP_BYPASS_FOR_TESTING:
            print(f"🧪 OTP_BYPASS_FOR_TESTING enabled: send_otp accepted for {phone}")
            return {
                "success": True,
                "message": f"OTP accepted for testing on {phone[-4:].rjust(len(phone), '*')}",
                "expires_in": OTP_EXPIRY_MINUTES * 60,
                "otp": "ANY",  # explicit testing marker for local/dev use
            }
        
        # ===== DEV MODE: Print to console =====
        if DEV_MODE:
            # Check rate limit
            existing = _otp_store.get(phone)
            if existing:
                last_sent = existing.get("sent_at")
                if last_sent and (datetime.utcnow() - last_sent).seconds < 60:
                    return {
                        "success": False,
                        "message": "Please wait 60 seconds before requesting another OTP"
                    }
            
            # Generate and store OTP locally
            otp = OTPService.generate_otp()
            _otp_store[phone] = {
                "otp": otp,
                "purpose": purpose,
                "sent_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES),
                "attempts": 0,
                "verified": False
            }
            
            # Print to console
            print(f"\n{'='*50}")
            print(f"📱 DEV MODE - OTP for {phone}")
            print(f"🔢 OTP: {otp}")
            print(f"⏱️  Valid for {OTP_EXPIRY_MINUTES} minutes")
            print(f"{'='*50}\n")
            
            return {
                "success": True,
                "message": f"OTP sent to {phone[-4:].rjust(len(phone), '*')}",
                "expires_in": OTP_EXPIRY_MINUTES * 60,
                "otp": otp  # Include OTP in dev mode for testing
            }
        
        # ===== PRODUCTION: Use Twilio Verify =====
        try:
            client = get_twilio_client()
            if not client:
                return {
                    "success": False,
                    "message": "SMS service not configured. Please contact support."
                }
            
            verification = client.verify.v2.services(
                TWILIO_VERIFY_SERVICE_SID
            ).verifications.create(
                to=phone,
                channel="sms"
            )
            
            print(f"📱 Twilio Verify sent to {phone}, status: {verification.status}")
            
            return {
                "success": True,
                "message": f"OTP sent to {phone[-4:].rjust(len(phone), '*')}",
                "expires_in": OTP_EXPIRY_MINUTES * 60
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Twilio Verify error: {error_msg}")
            
            # Handle common errors
            if "Invalid phone number" in error_msg or "is not a valid phone number" in error_msg:
                return {
                    "success": False,
                    "message": "Invalid phone number format"
                }
            elif "rate limit" in error_msg.lower():
                return {
                    "success": False,
                    "message": "Too many requests. Please wait before trying again."
                }
            
            return {
                "success": False,
                "message": "Failed to send OTP. Please try again."
            }
    
    @staticmethod
    def verify_otp(phone: str, otp: str, purpose: str = "registration") -> Dict:
        """
        Verify OTP for a phone number
        
        - DEV_MODE: Verifies against locally stored OTP
        - PRODUCTION: Uses Twilio Verify to check OTP
        
        Args:
            phone: Phone number
            otp: OTP entered by user
            purpose: Expected purpose
            
        Returns:
            Dictionary with verification result
        """
        phone = OTPService._format_phone(phone)

        # ===== TEST BYPASS MODE: accept any OTP value =====
        if OTP_BYPASS_FOR_TESTING:
            print(f"🧪 OTP_BYPASS_FOR_TESTING enabled: verify_otp accepted for {phone}")
            return {
                "success": True,
                "message": "Phone number verified successfully (testing bypass)"
            }
        
        # ===== DEV MODE: Verify locally =====
        if DEV_MODE:
            stored = _otp_store.get(phone)
            
            if not stored:
                return {
                    "success": False,
                    "message": "No OTP found. Please request a new one."
                }
            
            # Check purpose
            if stored.get("purpose") != purpose:
                return {
                    "success": False,
                    "message": "Invalid OTP request"
                }
            
            # Check expiry
            if datetime.utcnow() > stored.get("expires_at"):
                del _otp_store[phone]
                return {
                    "success": False,
                    "message": "OTP expired. Please request a new one."
                }
            
            # Check attempts
            if stored.get("attempts", 0) >= OTP_MAX_ATTEMPTS:
                del _otp_store[phone]
                return {
                    "success": False,
                    "message": "Too many attempts. Please request a new OTP."
                }
            
            # Increment attempts
            stored["attempts"] = stored.get("attempts", 0) + 1
            
            # Verify OTP
            if stored.get("otp") != otp:
                remaining = OTP_MAX_ATTEMPTS - stored["attempts"]
                return {
                    "success": False,
                    "message": f"Invalid OTP. {remaining} attempts remaining."
                }
            
            # Mark as verified
            stored["verified"] = True
            stored["verified_at"] = datetime.utcnow()
            stored["expires_at"] = datetime.utcnow() + timedelta(minutes=30)
            
            return {
                "success": True,
                "message": "Phone number verified successfully"
            }
        
        # ===== PRODUCTION: Use Twilio Verify =====
        try:
            client = get_twilio_client()
            if not client:
                return {
                    "success": False,
                    "message": "SMS service not configured"
                }
            
            verification_check = client.verify.v2.services(
                TWILIO_VERIFY_SERVICE_SID
            ).verification_checks.create(
                to=phone,
                code=otp
            )
            
            print(f"📱 Twilio Verify check for {phone}, status: {verification_check.status}")
            
            if verification_check.status == "approved":
                return {
                    "success": True,
                    "message": "Phone number verified successfully"
                }
            else:
                return {
                    "success": False,
                    "message": "Invalid OTP. Please try again."
                }
                
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Twilio Verify check error: {error_msg}")
            
            if "not found" in error_msg.lower() or "expired" in error_msg.lower():
                return {
                    "success": False,
                    "message": "OTP expired. Please request a new one."
                }
            elif "max check attempts" in error_msg.lower():
                return {
                    "success": False,
                    "message": "Too many attempts. Please request a new OTP."
                }
            
            return {
                "success": False,
                "message": "Verification failed. Please try again."
            }
    
    @staticmethod
    def is_verified(phone: str, purpose: str = "registration") -> bool:
        """
        Check if phone is verified for given purpose
        Note: Only works in DEV_MODE. In production, Twilio Verify handles state.
        """
        phone = OTPService._format_phone(phone)
        
        # In dev mode, check local store
        if DEV_MODE:
            stored = _otp_store.get(phone)
            
            if not stored:
                return False
            
            if stored.get("purpose") != purpose:
                return False
            
            if datetime.utcnow() > stored.get("expires_at"):
                return False
            
            return stored.get("verified", False)
        
        # In production, we don't maintain state - verification is one-time
        # The route should proceed after successful verify_otp()
        return True
    
    @staticmethod
    def clear_otp(phone: str):
        """Clear OTP after successful registration/action (dev mode only)"""
        phone = OTPService._format_phone(phone)
        if phone in _otp_store:
            del _otp_store[phone]


# For testing
if __name__ == "__main__":
    # Test OTP flow
    phone = "+919876543210"
    
    print("1. Sending OTP...")
    result = OTPService.send_otp(phone)
    print(f"Result: {result}")
    
    if result["success"]:
        otp = result.get("otp", input("Enter OTP: "))
        print(f"\n2. Verifying OTP: {otp}")
        verify_result = OTPService.verify_otp(phone, otp)
        print(f"Result: {verify_result}")
        
        print(f"\n3. Is verified: {OTPService.is_verified(phone)}")
