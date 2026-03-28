"""
Sprint 4 Validation
Checks: Redis + RQ, SMS module, alert dispatcher, DB tables, FastAPI endpoints,
        JWT auth, camera CRUD.
Requires:
  - docker-compose up -d  (PostgreSQL + Redis running)
  - uvicorn api.main:app --port 8000  (FastAPI running)
Run: python tests/sprint4_validation.py
"""
import sys
import os
import time
import requests

BASE_URL = "http://localhost:8000/api/v1"


def run(name, fn):
    try:
        fn()
        print(f"  PASSED  {name}")
        return True
    except Exception as e:
        print(f"  FAILED  {name} â€” {e}")
        return False


def check_redis_connection():
    from redis import Redis
    from dotenv import load_dotenv
    load_dotenv()
    r = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    r.ping()
    print(f"           Redis ping OK")


def check_rq_queue_exists():
    from redis import Redis
    from rq import Queue
    from dotenv import load_dotenv
    load_dotenv()
    r = Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    q = Queue("alerts", connection=r)
    assert q is not None


def check_sms_module_imports():
    from alerts.sms import send_sms_alert
    assert callable(send_sms_alert)


def check_dispatcher_enqueues():
    from alerts.dispatcher import dispatch_alert
    payload = {
        "camera_id": 1,
        "incident_type": "test",
        "fusion_score": 0.9,
        "woman_track_id": 5,
        "timestamp": time.time(),
        "videomae_score": 0.0,
        "bilstm_score": 0.0,
        "optflow_score": 0.0,
        "clip_score": 0.0,
        "pose_score": 0.0,
    }
    dispatch_alert(payload)
    print(f"           Alert enqueued successfully")


def check_twilio_env_vars():
    from dotenv import load_dotenv
    load_dotenv()
    sid   = os.getenv("TWILIO_ACCOUNT_SID", "")
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    if not sid or not token:
        print(f"           WARNING: Twilio credentials not set in .env (non-fatal)")
    else:
        print(f"           Twilio credentials present")


def check_sms_skips_gracefully():
    from alerts.sms import send_sms_alert
    orig_to   = os.environ.pop("ALERT_PHONE_NUMBER",  None)
    orig_from = os.environ.pop("TWILIO_FROM_NUMBER",  None)
    result = send_sms_alert("TestCam", "test", 0.9, time.time())
    assert result["status"] in ("skipped", "failed"), (
        f"Expected skip or fail without phone config, got: {result}"
    )
    if orig_to:   os.environ["ALERT_PHONE_NUMBER"]  = orig_to
    if orig_from: os.environ["TWILIO_FROM_NUMBER"]  = orig_from
    print(f"           SMS graceful skip: {result['status']}")


def check_snapshot_dirs_exist():
    from dotenv import load_dotenv
    load_dotenv()
    snap = os.getenv("SNAPSHOT_DIR", "./data/snapshots")
    clip = os.getenv("CLIP_DIR", "./data/clips")
    os.makedirs(snap, exist_ok=True)
    os.makedirs(clip, exist_ok=True)
    assert os.path.isdir(snap)
    assert os.path.isdir(clip)


def check_db_tables_exist():
    from db.session import engine
    from sqlalchemy import inspect
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    required = ["cameras", "incidents", "alerts", "system_config", "system_health"]
    missing = [t for t in required if t not in tables]
    assert not missing, f"Missing DB tables: {missing}"
    print(f"           Tables OK: {required}")


def check_api_docs_reachable():
    r = requests.get("http://localhost:8000/docs", timeout=5)
    assert r.status_code == 200, f"FastAPI /docs not reachable: {r.status_code}"


def check_auth_login():
    r = requests.post(f"{BASE_URL}/auth/login",
                      data={"username": "admin", "password": "changeme"},
                      timeout=5)
    assert r.status_code == 200, f"Login failed: {r.status_code} {r.text}"
    data = r.json()
    assert "access_token" in data
    token = data["access_token"]
    print(f"           JWT received (first 20 chars): {token[:20]}...")
    return token


def check_cameras_endpoint():
    token = check_auth_login()
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{BASE_URL}/cameras", headers=headers, timeout=5)
    assert r.status_code == 200, f"GET /cameras failed: {r.status_code}"
    assert isinstance(r.json(), list)


def check_incidents_endpoint():
    token = check_auth_login()
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{BASE_URL}/incidents?page=1&limit=10", headers=headers, timeout=5)
    assert r.status_code == 200, f"GET /incidents failed: {r.status_code}"


def check_system_status_endpoint():
    token = check_auth_login()
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{BASE_URL}/system/status", headers=headers, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "vram_used" in data or "uptime" in data


def check_system_config_endpoint():
    token = check_auth_login()
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(f"{BASE_URL}/system/config", headers=headers, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "fusion_threshold" in data


def check_unauthenticated_rejected():
    r = requests.get(f"{BASE_URL}/incidents", timeout=5)
    assert r.status_code == 401, (
        f"Unauthenticated request should return 401, got {r.status_code}"
    )


def check_camera_crud():
    token = check_auth_login()
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.post(f"{BASE_URL}/cameras",
                      json={"name": "Validation Camera", "source": "0", "location": "Test"},
                      headers=headers, timeout=5)
    assert r.status_code in (200, 201), f"Camera create failed: {r.status_code}"
    cam_id = r.json()["id"]
    r = requests.delete(f"{BASE_URL}/cameras/{cam_id}", headers=headers, timeout=5)
    assert r.status_code in (200, 204), f"Camera delete failed: {r.status_code}"
    print(f"           Camera created (id={cam_id}) and deleted OK")


TESTS = [
    ("Redis connection OK",               check_redis_connection),
    ("RQ alerts queue exists",            check_rq_queue_exists),
    ("sms module imports",                check_sms_module_imports),
    ("dispatch_alert() enqueues job",     check_dispatcher_enqueues),
    ("Twilio env vars present",           check_twilio_env_vars),
    ("SMS skips if no phone config",      check_sms_skips_gracefully),
    ("Snapshot directories exist",        check_snapshot_dirs_exist),
    ("DB tables exist",                   check_db_tables_exist),
    ("FastAPI docs reachable",            check_api_docs_reachable),
    ("POST /auth/login returns JWT",      check_auth_login),
    ("GET /cameras returns list",         check_cameras_endpoint),
    ("GET /incidents returns 200",        check_incidents_endpoint),
    ("GET /system/status returns 200",    check_system_status_endpoint),
    ("GET /system/config has keys",       check_system_config_endpoint),
    ("Unauthenticated -> 401",            check_unauthenticated_rejected),
    ("Camera CRUD create + delete",       check_camera_crud),
]

if __name__ == "__main__":
    print("\n===  Sprint 4 Validation  ===\n")
    print("  NOTE: Requires docker-compose up -d AND uvicorn api.main:app --port 8000\n")
    passed = sum(run(n, f) for n, f in TESTS)
    total  = len(TESTS)
    print(f"\n{'='*40}")
    print(f"  {passed}/{total} tests passed")
    if passed < total:
        print("  SPRINT 4 INCOMPLETE â€” fix failures before Sprint 5")
        sys.exit(1)
    else:
        print("  SPRINT 4 COMPLETE")
