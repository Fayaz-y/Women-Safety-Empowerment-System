"""
Women Safety AI — Frame Annotator
====================================
Draws rich visual annotations on video frames:
  • Per-person bounding box (colour-coded by threat level)
  • Gender label + confidence
  • Threat / distress score bar
  • Status badges: LONE, SURROUNDED, ALERT
  • Top info bar: camera, FPS, person count, alert count
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import cv2
import numpy as np

if TYPE_CHECKING:
    from core.tracking.tracker import TrackedPerson


# ── Colour palette ────────────────────────────────────────────────────────────
COL_GREEN  = (0, 200, 80)
COL_YELLOW = (0, 220, 255)
COL_RED    = (0, 0, 255)
COL_ORANGE = (0, 140, 255)
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0, 0, 0)
COL_BLUE   = (255, 160, 40)
COL_PURPLE = (200, 60, 200)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _threat_colour(score: float) -> tuple:
    """Green < 0.3, Yellow 0.3–0.6, Red > 0.6."""
    if score < 0.3:
        return COL_GREEN
    elif score < 0.6:
        return COL_YELLOW
    return COL_RED


def _draw_rounded_rect(
    img: np.ndarray, pt1: tuple, pt2: tuple, colour: tuple,
    thickness: int = -1, radius: int = 8, alpha: float = 0.7,
):
    """Draw a rounded semi-transparent rectangle."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _draw_badge(
    img: np.ndarray, text: str, x: int, y: int,
    bg_colour: tuple, text_colour: tuple = COL_WHITE,
):
    """Draw a small coloured badge with text."""
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    pad = 4
    cv2.rectangle(
        img, (x, y - th - pad * 2), (x + tw + pad * 2, y), bg_colour, -1,
    )
    cv2.putText(img, text, (x + pad, y - pad), FONT, 0.45, text_colour, 1, cv2.LINE_AA)


class FrameAnnotator:
    """Draws all visual annotations onto a video frame."""

    def __init__(self, camera_name: str = "Camera 1"):
        self.camera_name = camera_name
        self._alert_count = 0
        self._last_alert_frame = -999

    def set_alert_fired(self, frame_count: int):
        """Call when an alert fires to show the ALERT banner."""
        self._alert_count += 1
        self._last_alert_frame = frame_count

    def annotate(
        self,
        frame: np.ndarray,
        persons: list,
        threat_scores: Dict[int, float],
        lone_woman_ids: set,
        surrounded_ids: set,
        night_mode: bool,
        fps: int,
        frame_count: int,
    ) -> np.ndarray:
        """
        Draw all annotations on the frame.

        Args:
            frame:           BGR frame (modified in-place).
            persons:         List[TrackedPerson] — tracked people.
            threat_scores:   {track_id: fusion_score} for females.
            lone_woman_ids:  Set of track_ids flagged as lone women.
            surrounded_ids:  Set of track_ids flagged as surrounded.
            night_mode:      True if frame brightness is low.
            fps:             Current FPS.
            frame_count:     Current frame number.
        """
        h, w = frame.shape[:2]

        # ── Per-person annotations ─────────────────────────────────────
        for person in persons:
            tid = person.track_id
            bbox = person.bbox
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])

            gender = getattr(person, "gender", "unknown")
            gender_conf = getattr(person, "gender_conf", 0.0)
            threat = threat_scores.get(tid, 0.0)

            # Colour based on threat level for females, blue for males
            if gender == "female":
                colour = _threat_colour(threat)
            elif gender == "male":
                colour = COL_BLUE
            else:
                colour = (150, 150, 150)

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

            # Gender + ID label
            gender_sym = "♀" if gender == "female" else ("♂" if gender == "male" else "?")
            label = f"ID:{tid} {gender_sym}{gender.upper()} {gender_conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, FONT, 0.5, 1)

            # Label background
            cv2.rectangle(frame, (x1, y1 - lh - 10), (x1 + lw + 6, y1), colour, -1)
            cv2.putText(
                frame, label, (x1 + 3, y1 - 5),
                FONT, 0.5, COL_WHITE, 1, cv2.LINE_AA,
            )

            # Threat score bar (for females only)
            if gender == "female" and threat > 0.0:
                bar_w = x2 - x1
                bar_h = 6
                bar_y = y2 + 4
                # Background
                cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_h), (60, 60, 60), -1)
                # Filled portion
                fill_w = int(bar_w * min(threat, 1.0))
                bar_colour = _threat_colour(threat)
                cv2.rectangle(frame, (x1, bar_y), (x1 + fill_w, bar_y + bar_h), bar_colour, -1)
                # Score text
                cv2.putText(
                    frame, f"{threat:.0%}",
                    (x1, bar_y + bar_h + 14), FONT, 0.4, bar_colour, 1, cv2.LINE_AA,
                )

            # Status badges (below bbox on the left)
            badge_y = y2 + 30
            if tid in lone_woman_ids:
                _draw_badge(frame, "LONE WOMAN", x1, badge_y, COL_ORANGE)
                badge_y += 20
            if tid in surrounded_ids:
                _draw_badge(frame, "SURROUNDED", x1, badge_y, COL_PURPLE)
                badge_y += 20

        # ── Alert banner (flashes for 60 frames after an alert) ────────
        if (frame_count - self._last_alert_frame) < 60:
            banner_h = 40
            _draw_rounded_rect(frame, (0, 0), (w, banner_h), COL_RED, alpha=0.8)
            alert_text = f"!! THREAT DETECTED — ALERT #{self._alert_count} SENT !!"
            (tw, th), _ = cv2.getTextSize(alert_text, FONT_BOLD, 0.7, 2)
            tx = (w - tw) // 2
            cv2.putText(
                frame, alert_text, (tx, 28),
                FONT_BOLD, 0.7, COL_WHITE, 2, cv2.LINE_AA,
            )

        # ── Bottom info bar ───────────────────────────────────────────
        bar_h = 32
        bar_y = h - bar_h
        _draw_rounded_rect(frame, (0, bar_y), (w, h), COL_BLACK, alpha=0.6)

        n_persons = len(persons)
        n_females = sum(1 for p in persons if getattr(p, "gender", "") == "female")
        n_males = sum(1 for p in persons if getattr(p, "gender", "") == "male")

        night_str = " | NIGHT" if night_mode else ""
        info = (
            f"{self.camera_name} | FPS: {fps} | "
            f"Persons: {n_persons} (F:{n_females} M:{n_males}) | "
            f"Alerts: {self._alert_count}{night_str}"
        )
        cv2.putText(
            frame, info, (10, h - 10),
            FONT, 0.5, COL_WHITE, 1, cv2.LINE_AA,
        )

        return frame
