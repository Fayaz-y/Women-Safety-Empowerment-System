"""
Women Safety AI — WebSocket: Real-Time Alert Push
====================================================
Maintains a set of connected dashboard clients and broadcasts
alert payloads when incidents are created.
"""

import asyncio
from typing import Set

from fastapi import WebSocket, WebSocketDisconnect

_ws_clients: Set[WebSocket] = set()


async def endpoint(websocket: WebSocket):
    """WebSocket endpoint: /ws/alerts"""
    await websocket.accept()
    _ws_clients.add(websocket)
    try:
        while True:
            # Keep the connection alive — client sends keep-alive pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        _ws_clients.discard(websocket)


async def broadcast_alert(payload: dict):
    """
    Broadcast an alert payload to all connected WebSocket clients.

    Called from the alert dispatcher after saving an incident to DB.
    """
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def broadcast_alert_sync(payload: dict):
    """
    Synchronous wrapper for ``broadcast_alert``.

    Called from the RQ worker process (which runs synchronously).
    Creates a new event loop if necessary.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule the coroutine on the running loop
            asyncio.ensure_future(broadcast_alert(payload))
        else:
            loop.run_until_complete(broadcast_alert(payload))
    except RuntimeError:
        # No event loop exists in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(broadcast_alert(payload))
        finally:
            loop.close()
