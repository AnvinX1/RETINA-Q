"""
Patient Routes — CRUD endpoints for patient management.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from loguru import logger

from app.database import get_db
from app.db_models.patient import Patient
from app.db_models.scan import Scan
from app.schemas.responses import (
    PatientCreate,
    PatientUpdate,
    PatientResponse,
    PatientListItem,
    PaginatedResponse,
)


router = APIRouter(prefix="/api/patients", tags=["Patients"])


@router.post("", response_model=PatientResponse, status_code=201)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient record."""
    existing = db.query(Patient).filter(Patient.patient_id == payload.patient_id).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Patient ID '{payload.patient_id}' already exists")

    patient = Patient(**payload.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    logger.info(f"Patient created — id={patient.id}, patient_id={patient.patient_id}")
    return patient


@router.get("", response_model=PaginatedResponse)
def list_patients(
    search: str | None = Query(None, description="Search by name or patient_id"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List patients with optional search and pagination."""
    query = db.query(Patient)

    if search:
        like_pattern = f"%{search}%"
        query = query.filter(
            (Patient.name.ilike(like_pattern)) | (Patient.patient_id.ilike(like_pattern))
        )

    total = query.count()
    patients = query.order_by(Patient.created_at.desc()).offset((page - 1) * per_page).limit(per_page).all()

    # Attach scan counts
    patient_ids = [p.id for p in patients]
    scan_counts = {}
    if patient_ids:
        rows = (
            db.query(Scan.patient_id, func.count(Scan.id))
            .filter(Scan.patient_id.in_(patient_ids))
            .group_by(Scan.patient_id)
            .all()
        )
        scan_counts = {pid: cnt for pid, cnt in rows}

    items = []
    for p in patients:
        item = PatientListItem(
            id=p.id,
            patient_id=p.patient_id,
            name=p.name,
            age=p.age,
            gender=p.gender,
            scan_count=scan_counts.get(p.id, 0),
            created_at=p.created_at,
        )
        items.append(item)

    return PaginatedResponse(items=[i.model_dump() for i in items], total=total, page=page, per_page=per_page)


@router.get("/{patient_db_id}", response_model=PatientResponse)
def get_patient(patient_db_id: int, db: Session = Depends(get_db)):
    """Get patient detail with full scan history."""
    patient = db.query(Patient).filter(Patient.id == patient_db_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/{patient_db_id}", response_model=PatientResponse)
def update_patient(patient_db_id: int, payload: PatientUpdate, db: Session = Depends(get_db)):
    """Update patient fields."""
    patient = db.query(Patient).filter(Patient.id == patient_db_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(patient, key, value)

    db.commit()
    db.refresh(patient)
    logger.info(f"Patient updated — id={patient.id}")
    return patient


@router.delete("/{patient_db_id}", status_code=204)
def delete_patient(patient_db_id: int, db: Session = Depends(get_db)):
    """Delete a patient and all associated scans."""
    patient = db.query(Patient).filter(Patient.id == patient_db_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    db.delete(patient)
    db.commit()
    logger.info(f"Patient deleted — id={patient_db_id}")
