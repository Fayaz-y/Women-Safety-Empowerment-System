"""
Women Safety AI — Video Clip Saver
===================================
Saves annotated video frames to MP4 when an incident is triggered.
"""

import os
import cv2
from datetime import datetime
from typing import Optional


class ClipSaver:
    """Buffers annotated frames and saves them to video on incident trigger."""

    def __init__(self, clip_dir: str = "./data/clips", buffer_size: int = 300):
        """
        Args:
            clip_dir: Directory to save clips
            buffer_size: Number of frames to buffer (300 frames ≈ 20 seconds at 15 FPS)
        """
        self.clip_dir = clip_dir
        self.buffer_size = buffer_size
        self.frame_buffer = []
        self.fps = 15.0
        self.frame_size = None
        os.makedirs(clip_dir, exist_ok=True)

    def add_frame(self, frame):
        """Add annotated frame to buffer."""
        if self.frame_size is None and frame is not None:
            self.frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
        
        if frame is not None:
            self.frame_buffer.append(frame.copy())
            # Keep buffer size limited
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)

    def save_clip(
        self, 
        incident_type: str,
        camera_id: int,
        fusion_score: float,
        gender: Optional[str] = None,
        woman_track_id: Optional[int] = None,
    ) -> Optional[str]:
        """
        Save buffered frames as MP4 video with incident metadata in filename.
        
        Returns:
            Path to saved video file, or None if save failed.
        """
        if not self.frame_buffer or self.frame_size is None:
            return None

        # Create filename with incident details
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gender_str = f"_{gender}" if gender else ""
        score_str = f"_{fusion_score:.2f}".replace(".", "p")
        filename = f"incident_{incident_type}{gender_str}_{score_str}_{timestamp}.mp4"
        filepath = os.path.join(self.clip_dir, filename)

        try:
            # Setup VideoWriter with H.264 codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filepath, fourcc, self.fps, self.frame_size)

            if not out.isOpened():
                print(f"[ClipSaver] ERROR: Could not open VideoWriter for {filepath}")
                return None

            # Write all buffered frames
            for frame in self.frame_buffer:
                out.write(frame)

            out.release()
            print(f"[ClipSaver] ✓ Saved clip: {filename}")
            return filepath

        except Exception as e:
            print(f"[ClipSaver] ERROR saving clip: {e}")
            return None

    def clear(self):
        """Clear the frame buffer."""
        self.frame_buffer.clear()
