"""
Women Safety AI — Camera CRUD Routes
======================================
Full CRUD for camera records. All endpoints require JWT auth.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.schemas import CameraCreate, CameraSchema, CameraUpdate
from db.models import Camera
from db.session import get_db

router = APIRouter()


@router.get("/", response_model=list[CameraSchema])
def list_cameras(
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
):
    """Return all registered cameras."""
    return db.query(Camera).all()


@router.post("/", response_model=CameraSchema, status_code=201)
def create_camera(
    body: CameraCreate,
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
):
    """Register a new camera source."""
    cam = Camera(name=body.name, source=body.source, location=body.location)
    db.add(cam)
    db.commit()
    db.refresh(cam)
    return cam


@router.patch("/{camera_id}", response_model=CameraSchema)
def update_camera(
    camera_id: int,
    body: CameraUpdate,
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
):
    """Update fields on an existing camera."""
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(cam, field, value)
    db.commit()
    db.refresh(cam)
    return cam


@router.delete("/{camera_id}")
def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    _user: str = Depends(get_current_user),
):
    """Remove a camera record."""
    cam = db.query(Camera).filter(Camera.id == camera_id).first()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    db.delete(cam)
    db.commit()
    return {"ok": True}
