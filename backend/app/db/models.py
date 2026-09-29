"""
SQLAlchemy Spatial and Telemetry Models for GeoDelta
Implements persistent storage for detected changes and cryptographic audit trails.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, Text, DateTime, JSON
from app.db.session import Base


class DetectedChange(Base):
    """
    Persistent record of a vectorized change feature.
    Corresponds to detected_changes table in PostgreSQL/PostGIS.
    """
    __tablename__ = "detected_changes"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_id = Column(String(64), index=True, nullable=False)
    tactical_class = Column(String(128), nullable=False)
    confidence = Column(Float, nullable=False)
    area_sq_m = Column(Float, nullable=False)
    area_hectares = Column(Float, nullable=False)
    centroid_lat = Column(Float, nullable=False)
    centroid_lon = Column(Float, nullable=False)
    centroid_mgrs = Column(String(32), nullable=False)
    geom_geojson = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<DetectedChange(id={self.id}, task_id={self.task_id}, class={self.tactical_class}, area_sq_m={self.area_sq_m})>"


class AuditLog(Base):
    """
    Cryptographic chain-of-custody and execution audit telemetry record.
    Corresponds to audit_logs table.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), index=True, nullable=False)
    t1_sha256 = Column(String(64), nullable=False)
    t2_sha256 = Column(String(64), nullable=False)
    prompt = Column(Text, nullable=False)
    negative_suppression = Column(Text, nullable=True)
    polygons_detected = Column(Integer, nullable=False)
    total_area_sq_m = Column(Float, nullable=False)
    execution_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        return f"<AuditLog(task_id={self.task_id}, polygons={self.polygons_detected}, total_area_sq_m={self.total_area_sq_m})>"
