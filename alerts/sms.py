"""
Women Safety AI — Twilio SMS Sender
=====================================
Lazy-initialises a Twilio REST client on first use.
Gracefully skips if phone numbers are not configured.
"""

import os
from datetime import datetime

_client = None


def _get_client():
    """Lazy-init the Twilio client."""
    global _client
    if _client is None:
        from twilio.rest import Client

        sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        token = os.getenv("TWILIO_AUTH_TOKEN", "")
        _client = Client(sid, token)
    return _client


def send_sms_alert(
    camera_name: str,
    incident_type: str,
    confidence: float,
    timestamp: float,
) -> dict:
    """
    Send an SMS alert via Twilio.

    Returns a dict with ``status`` key:
      - ``"skipped"``  — phone numbers not configured
      - ``"sent"``     — SMS dispatched (includes ``sid``)
      - ``"failed"``   — exception occurred (includes ``error``)
    """
    to_number = os.getenv("ALERT_PHONE_NUMBER", "")
    from_number = os.getenv("TWILIO_FROM_NUMBER", "")

    if not to_number or not from_number:
        return {"status": "skipped", "reason": "phone numbers not configured"}

    dt_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    body = (
        f"SAFETY ALERT\n"
        f"Type: {incident_type.upper()}\n"
        f"Camera: {camera_name}\n"
        f"Confidence: {confidence:.0%}\n"
        f"Time: {dt_str}"
    )

    try:
        message = _get_client().messages.create(
            body=body,
            from_=from_number,
            to=to_number,
        )
        return {"status": "sent", "sid": message.sid}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
