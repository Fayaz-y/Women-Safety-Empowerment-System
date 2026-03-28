"""
Women Safety AI — Incident Query Routes
=========================================
Paginated, filterable incident listing and acknowledgement.
All endpoints require JWT auth.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.schemas import IncidentSchema, PaginatedIncidents
from db.models import Incident
from db.session import get_db

router = APIRouter()


@router.get("/", response_model=PaginatedIncidents)
def list_incidents(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    camera_id: Optional[int] = None,
    incident_type: Optional[str] = None,
    acknowledged: Optional[bool] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
):
    """Return a paginated, filtered list of incidents."""
    q = db.query(Incident)

    if camera_id is not None:
        q = q.filter(Incident.camera_id == camera_id)
    if incident_type is not None:
        q = q.filter(Incident.incident_type == incident_type)
    if acknowledged is not None:
        q = q.filter(Incident.acknowledged == acknowledged)
    if date_from is not None:
        q = q.filter(Incident.created_at >= date_from)
    if date_to is not None:
        q = q.filter(Incident.created_at <= date_to)

    total = q.count()
    items = (
        q.order_by(Incident.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    return {
        "total": total,
        "items": items,
        "page": page,
        "limit": limit,
    }


@router.get("/{incident_id}", response_model=IncidentSchema)
def get_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
):
    """Return full incident detail with all score fields."""
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    return inc


@router.patch("/{incident_id}/acknowledge", response_model=IncidentSchema)
def acknowledge_incident(
    incident_id: int,
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
):
    """Mark an incident as acknowledged."""
    inc = db.query(Incident).filter(Incident.id == incident_id).first()
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found")
    inc.acknowledged = True
    inc.acknowledged_at = datetime.utcnow()
    db.commit()
    db.refresh(inc)
    return inc
