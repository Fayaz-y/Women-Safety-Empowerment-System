"""
Women Safety AI — Thread-Safe Camera Stream
=============================================
One instance per physical camera or video file. Reads frames in a
background daemon thread and always exposes the latest frame — the
inference loop never blocks waiting for a frame.

Supports:
  - Integer source (0, 1, …) → USB / built-in webcam
  - String source ("path/to/video.mp4") → video file (loops automatically)

Usage:
    cam = CameraStream(source=0, camera_id=0).start()
    frame = cam.read()   # returns latest frame (or None)
    cam.stop()
"""

import threading
import time
from typing import Optional, Union

import cv2
import numpy as np


class CameraStream:
    """Thread-safe camera/video reader with sliding-window FPS calculation."""

    def __init__(
        self,
        source: Union[int, str] = 0,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
    ):
        self.source = source
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.is_file = isinstance(source, str)

        # Open capture
        self.cap = cv2.VideoCapture(source)
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # discard stale frames

        # State
        self.frame: Optional[np.ndarray] = None
        self.running: bool = False
        self.fps: int = 0

        # Thread safety
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._frame_times: list = []

    def start(self) -> "CameraStream":
        """Start the background frame-reading thread."""
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Camera/video source {self.source} could not be opened. "
                "Check if the device is connected or the file path is correct."
            )
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        # Wait for the first frame to avoid returning None immediately after start
        timeout = time.time() + 5.0
        while self.frame is None and time.time() < timeout:
            time.sleep(0.05)

        return self

    def _loop(self) -> None:
        """Background loop — continuously reads the latest frame."""
        # For video files, cap the playback speed to ~original FPS
        target_delay = 0.0
        if self.is_file:
            file_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if file_fps and file_fps > 0:
                target_delay = 1.0 / file_fps

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:
                    # Loop video file: seek back to the beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # Live camera glitch — brief pause and retry
                    time.sleep(0.01)
                    continue

            with self._lock:
                self.frame = frame

            # Sliding-window FPS
            now = time.time()
            self._frame_times.append(now)
            # Keep only timestamps within the last 1 second
            self._frame_times = [
                t for t in self._frame_times if now - t <= 1.0
            ]
            self.fps = len(self._frame_times)

            # Throttle video file playback to its native FPS
            if target_delay > 0:
                time.sleep(target_delay)

    def read(self) -> Optional[np.ndarray]:
        """Return a copy of the latest frame (thread-safe), or None."""
        with self._lock:
            if self.frame is not None:
                return self.frame.copy()
            return None

    def stop(self) -> None:
        """Stop the background thread and release the camera."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()

    @property
    def is_open(self) -> bool:
        """True if the camera is running and the capture device is open."""
        return self.running and self.cap.isOpened()

