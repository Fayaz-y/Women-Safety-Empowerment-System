"""
Women Safety AI — Pydantic API Schemas
========================================
Request / response models for the FastAPI REST API.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# ── Camera schemas ──────────────────────────────────────────────────


class CameraCreate(BaseModel):
    name: str
    source: str
    location: Optional[str] = None


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    source: Optional[str] = None
    location: Optional[str] = None
    is_active: Optional[bool] = None


class CameraSchema(BaseModel):
    id: int
    name: str
    source: str
    location: Optional[str] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── Incident schemas ────────────────────────────────────────────────


class IncidentSchema(BaseModel):
    id: int
    camera_id: Optional[int] = None
    track_id: Optional[int] = None
    incident_type: str
    confidence: float
    fusion_score: Optional[float] = None
    videomae_score: Optional[float] = None
    bilstm_score: Optional[float] = None
    optflow_score: Optional[float] = None
    clip_score: Optional[float] = None
    pose_score: Optional[float] = None
    snapshot_path: Optional[str] = None
    clip_path: Optional[str] = None
    acknowledged: bool
    acknowledged_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PaginatedIncidents(BaseModel):
    total: int
    items: list[IncidentSchema]
    page: int
    limit: int


# ── Token schema ────────────────────────────────────────────────────


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
