"""
Women Safety AI — Thread-Safe Camera Stream
=============================================
One instance per physical camera. Reads frames in a background daemon
thread and always exposes the latest frame — the inference loop never
blocks waiting for a frame.

Usage:
    cam = CameraStream(source=0, camera_id=0).start()
    frame = cam.read()   # returns latest frame (or None)
    cam.stop()
"""

import threading
import time
from typing import Optional

import cv2
import numpy as np


class CameraStream:
    """Thread-safe USB camera reader with sliding-window FPS calculation."""

    def __init__(
        self,
        source: int = 0,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
    ):
        self.source = source
        self.camera_id = camera_id
        self.width = width
        self.height = height

        # Open capture
        self.cap = cv2.VideoCapture(source)
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
                f"Camera source {self.source} could not be opened. "
                "Check if the device is connected."
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
        while self.running:
            ret, frame = self.cap.read()
            if ret:
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
            else:
                # Brief pause to avoid busy-waiting on failed reads
                time.sleep(0.01)

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
