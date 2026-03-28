"""
Women Safety AI — SQLAlchemy ORM Models
========================================
All database tables for the Women Safety system.
Single source of truth for the PostgreSQL schema.
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Camera(Base):
    """Registered camera source."""

    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    name = Column(String(100), nullable=False)
    source = Column(String(255), nullable=False)
    location = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    incidents = relationship("Incident", back_populates="camera")
    health_logs = relationship("SystemHealth", back_populates="camera")


class Incident(Base):
    """Detected safety incident record."""

    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    track_id = Column(Integer, nullable=True)
    incident_type = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    fusion_score = Column(Float, nullable=True)
    videomae_score = Column(Float, nullable=True)
    bilstm_score = Column(Float, nullable=True)
    optflow_score = Column(Float, nullable=True)
    clip_score = Column(Float, nullable=True)
    pose_score = Column(Float, nullable=True)
    snapshot_path = Column(String(500), nullable=True)
    clip_path = Column(String(500), nullable=True)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    camera = relationship("Camera", back_populates="incidents")
    alerts = relationship("Alert", back_populates="incident", cascade="all, delete-orphan")


class Alert(Base):
    """Alert dispatch record (SMS, etc.)."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(Integer, ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False)
    channel = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False)
    sent_at = Column(DateTime, nullable=True)
    error_msg = Column(Text, nullable=True)

    incident = relationship("Incident", back_populates="alerts")


class SystemConfig(Base):
    """Key-value system configuration."""

    __tablename__ = "system_config"

    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SystemHealth(Base):
    """Per-camera health telemetry snapshots."""

    __tablename__ = "system_health"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=True)
    fps = Column(Float, nullable=True)
    vram_used = Column(Float, nullable=True)
    cpu_percent = Column(Float, nullable=True)
    logged_at = Column(DateTime, default=datetime.utcnow, index=True)

    camera = relationship("Camera", back_populates="health_logs")
