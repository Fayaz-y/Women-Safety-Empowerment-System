"""
Women Safety AI — Seed Default System Config
===============================================
Inserts the default system_config rows. Idempotent — running
multiple times will not create duplicates (uses upsert logic).

Run::

    python scripts/seed_config.py
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.session import SessionLocal
from db.models import SystemConfig

DEFAULTS = {
    "fusion_threshold": "0.75",
    "proximity_radius_px": "150",
    "isolation_seconds": "5.0",
    "alert_cooldown_seconds": "30",
    "weight_videomae": "0.30",
    "weight_bilstm": "0.25",
    "weight_optflow": "0.20",
    "weight_clip": "0.15",
    "weight_pose": "0.10",
    "incident_retention_days": "30",
    "alert_phone_number": "",
}


def seed():
    db = SessionLocal()
    try:
        for key, value in DEFAULTS.items():
            existing = db.query(SystemConfig).filter(SystemConfig.key == key).first()
            if existing is None:
                db.add(SystemConfig(key=key, value=value))
                print(f"  [+] Inserted: {key} = {value}")
            else:
                print(f"  [=] Already exists: {key} = {existing.value}")
        db.commit()
        print("\n✅ Seed complete.")
    except Exception as e:
        db.rollback()
        print(f"\n❌ Seed failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    print("Seeding system_config defaults …")
    seed()
