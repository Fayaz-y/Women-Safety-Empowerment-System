"""
Sprint 2 Validation
Checks: PoseEstimator, distress analysis, ProximityEngine lone-woman timer,
        proximity violation event emission.
Run: python tests/sprint2_validation.py
"""
import sys
import time
import numpy as np


def run(name, fn):
    try:
        fn()
        print(f"  PASSED  {name}")
        return True
    except Exception as e:
        print(f"  FAILED  {name} â€” {e}")
        return False


def check_pose_estimator_loads():
    from core.pose.estimator import PoseEstimator
    pe = PoseEstimator()
    assert pe.model is not None


def check_pose_returns_list():
    from core.pose.estimator import PoseEstimator
    pe = PoseEstimator()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    result = pe.estimate(frame)
    assert isinstance(result, list)


def check_distress_flags_structure():
    from core.pose.estimator import PoseEstimator
    pe = PoseEstimator()
    kp = np.zeros((17, 3), dtype=np.float32)
    flags, score = pe._analyse_distress(kp)
    assert isinstance(flags, dict)
    assert set(flags.keys()) == {"raised_arms", "covering_face", "falling", "hunched"}
    assert 0.0 <= score <= 1.0


def check_distress_raised_arms():
    from core.pose.estimator import PoseEstimator
    pe = PoseEstimator()
    kp = np.zeros((17, 3), dtype=np.float32)
    # Left shoulder at y=200, left wrist at y=100 (wrist ABOVE shoulder in image)
    kp[5] = [100, 200, 0.9]  # left shoulder
    kp[9] = [100, 100, 0.9]  # left wrist â€” above shoulder
    flags, score = pe._analyse_distress(kp)
    assert flags["raised_arms"] is True, "raised_arms should be True when wrist is above shoulder"
    assert score > 0.0


def check_proximity_no_violation():
    from core.proximity.engine import ProximityEngine
    from core.tracking.tracker import TrackedPerson
    engine = ProximityEngine(radius_px=150)
    woman = TrackedPerson(track_id=1, bbox=[100, 100, 200, 400])
    woman.gender = "female"
    male = TrackedPerson(track_id=2, bbox=[600, 100, 700, 400])
    male.gender = "male"
    events = engine.update([woman, male])
    assert events == [], f"Expected no events, got: {events}"


def check_proximity_lone_woman_timer():
    from core.proximity.engine import ProximityEngine
    from core.tracking.tracker import TrackedPerson
    engine = ProximityEngine(radius_px=150, isolation_threshold=0.1)
    woman = TrackedPerson(track_id=10, bbox=[400, 100, 500, 400])
    woman.gender = "female"
    for _ in range(5):
        engine.update([woman])
        time.sleep(0.03)
    assert woman.isolation_seconds >= 0.1, (
        f"isolation_seconds not accumulating: {woman.isolation_seconds}"
    )


def check_proximity_violation_event():
    from core.proximity.engine import ProximityEngine
    from core.tracking.tracker import TrackedPerson
    engine = ProximityEngine(radius_px=300, isolation_threshold=0.05)
    woman = TrackedPerson(track_id=20, bbox=[400, 200, 500, 500])
    woman.gender = "female"
    # Build up lone-woman status first
    for _ in range(3):
        engine.update([woman])
        time.sleep(0.05)
    assert woman.is_lone_woman is True, "Woman should be lone after isolation period"
    # Now bring a male within radius
    male = TrackedPerson(track_id=21, bbox=[420, 200, 520, 500])
    male.gender = "male"
    events = engine.update([woman, male])
    assert len(events) > 0, "Expected proximity_violation event"
    assert events[0]["type"] == "proximity_violation"
    print(f"           Event: {events[0]['type']} dist={events[0]['distance_px']:.0f}px")


TESTS = [
    ("PoseEstimator loads",               check_pose_estimator_loads),
    ("estimate() returns list",           check_pose_returns_list),
    ("distress_flags structure correct",  check_distress_flags_structure),
    ("raised_arms flag fires correctly",  check_distress_raised_arms),
    ("No event when male is far away",    check_proximity_no_violation),
    ("Isolation timer accumulates",       check_proximity_lone_woman_timer),
    ("Proximity violation event fires",   check_proximity_violation_event),
]

if __name__ == "__main__":
    print("\n===  Sprint 2 Validation  ===\n")
    passed = sum(run(n, f) for n, f in TESTS)
    total  = len(TESTS)
    print(f"\n{'='*40}")
    print(f"  {passed}/{total} tests passed")
    if passed < total:
        print("  SPRINT 2 INCOMPLETE â€” fix failures before Sprint 3")
        sys.exit(1)
    else:
        print("  SPRINT 2 COMPLETE")
