import os
from sqlalchemy.orm import Session
from db.session import SessionLocal, engine
from db.models import Camera, Base

Base.metadata.create_all(bind=engine)

def seed_cameras():
    db = SessionLocal()
    try:
        # Check if camera 1 exists
        cam1 = db.query(Camera).filter(Camera.id == 1).first()
        if not cam1:
            cam1 = Camera(id=1, name="Main Camera", source="0", location="Entry Point")
            db.add(cam1)

        # Check if camera 2 exists
        cam2 = db.query(Camera).filter(Camera.id == 2).first()
        if not cam2:
            cam2 = Camera(id=2, name="Secondary Camera", source="1", location="Corridor")
            db.add(cam2)

        db.commit()
        print("Cameras seeded successfully in database.")
    except Exception as e:
        db.rollback()
        print(f"Error seeding cameras: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_cameras()
