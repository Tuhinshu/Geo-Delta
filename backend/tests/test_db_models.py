"""
Unit Tests for Database Models and Session (DetectedChange & AuditLog).
"""

import uuid
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.session import Base
from app.db.models import DetectedChange, AuditLog


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_detected_change_model_crud(in_memory_db):
    """Verifies creation and retrieval of DetectedChange spatial record."""
    feature_id = str(uuid.uuid4())
    task_id = "test-task-123"

    change = DetectedChange(
        id=feature_id,
        task_id=task_id,
        tactical_class="Surface Grading",
        confidence=0.9123,
        area_sq_m=12500.5,
        area_hectares=1.25,
        centroid_lat=25.123456,
        centroid_lon=75.654321,
        centroid_mgrs="43R EH 12345 67890",
        geom_geojson={"type": "MultiPolygon", "coordinates": []}
    )
    in_memory_db.add(change)
    in_memory_db.commit()

    retrieved = in_memory_db.query(DetectedChange).filter_by(id=feature_id).first()
    assert retrieved is not None
    assert retrieved.task_id == task_id
    assert retrieved.tactical_class == "Surface Grading"
    assert retrieved.area_sq_m == 12500.5
    assert retrieved.centroid_mgrs == "43R EH 12345 67890"


def test_audit_log_model_crud(in_memory_db):
    """Verifies creation and retrieval of cryptographic AuditLog record."""
    task_id = "audit-task-456"

    log = AuditLog(
        task_id=task_id,
        t1_sha256="abc123hash_t1",
        t2_sha256="def456hash_t2",
        prompt="Identify newly graded perimeter roads",
        negative_suppression="seasonal vegetation",
        polygons_detected=4,
        total_area_sq_m=48200.0,
        execution_time_ms=342.1
    )
    in_memory_db.add(log)
    in_memory_db.commit()

    retrieved = in_memory_db.query(AuditLog).filter_by(task_id=task_id).first()
    assert retrieved is not None
    assert retrieved.polygons_detected == 4
    assert retrieved.t1_sha256 == "abc123hash_t1"
    assert retrieved.execution_time_ms == 342.1
