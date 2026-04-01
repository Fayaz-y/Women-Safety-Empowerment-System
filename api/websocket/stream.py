"""
Women Safety AI — WebSocket: Live Annotated Frame Stream
==========================================================
Streams JPEG-encoded annotated frames from a running
``PipelineEngine`` to connected WebSocket clients.
"""

import asyncio
import base64
from typing import Dict

import cv2
from fastapi import WebSocket, WebSocketDisconnect

# Global engine registry — populated when engines start.
ENGINES: Dict[int, object] = {}


async def endpoint(websocket: WebSocket, camera_id: int):
    """WebSocket endpoint: /ws/stream/{camera_id}"""
    await websocket.accept()

    engine = ENGINES.get(camera_id)
    if engine is None:
        await websocket.send_json(
            {"error": f"No engine running for camera {camera_id}"}
        )
        await websocket.close()
        return

    try:
        while True:
            frame = engine.get_annotated_frame()
            if frame is None:
                await asyncio.sleep(0.1)  # Increased from 0.05
                continue

            _, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            await websocket.send_json(
                {"camera_id": camera_id, "frame": b64}
            )
            await asyncio.sleep(1 / 10)  # Reduced to 10 FPS
    except WebSocketDisconnect:
        pass  # client disconnected
