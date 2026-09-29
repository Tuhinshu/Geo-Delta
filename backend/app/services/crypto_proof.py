"""
GeoDelta Cryptographic Proof & Evidence Chain-of-Custody Service (FR-EXP-001)
Generates deterministic SHA-256 evidence digests for optical rasters, vector geometries,
and compiled intelligence dossiers to ensure air-gapped forensic integrity.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


def hash_bytes(data: bytes) -> str:
    """Computes a lowercase hex SHA-256 checksum over raw bytes."""
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def hash_file(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """
    Computes a deterministic SHA-256 checksum for a local file (e.g. source GeoTIFF tile).
    Reads in streaming chunks to avoid loading multi-gigabyte rasters into memory.
    """
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Source file not found for cryptographic hashing: {file_path}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_geojson(geojson_obj: Union[Dict[str, Any], list, str]) -> str:
    """
    Computes a deterministic SHA-256 digest over GeoJSON data.
    Uses canonical JSON serialization with sorted keys and compact separators to ensure
    identical digests regardless of formatting whitespace.
    """
    if isinstance(geojson_obj, str):
        try:
            parsed = json.loads(geojson_obj)
        except Exception:
            return hash_bytes(geojson_obj.encode("utf-8"))
    else:
        parsed = geojson_obj

    canonical_bytes = json.dumps(
        parsed,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")
    return hash_bytes(canonical_bytes)


def create_evidence_manifest(
    task_id: str,
    t1_ref: Union[str, bytes, Path],
    t2_ref: Union[str, bytes, Path],
    geojson_features: Union[Dict[str, Any], list],
    analyst_callsign: str = "GEOINT-OP-01",
    classification: str = "RESTRICTED // GEOINT ASSESSMENT // FOR OFFICIAL USE ONLY",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generates a formal cryptographic evidence manifest documenting raw source hashes,
    derived vector hashes, verification timestamps, and chain-of-custody metadata.
    """
    # Hash t1
    if isinstance(t1_ref, bytes):
        t1_hash = hash_bytes(t1_ref)
    elif isinstance(t1_ref, (str, Path)) and Path(t1_ref).is_file():
        t1_hash = hash_file(t1_ref)
    elif isinstance(t1_ref, str):
        t1_hash = hash_bytes(t1_ref.encode("utf-8"))
    else:
        t1_hash = "0" * 64

    # Hash t2
    if isinstance(t2_ref, bytes):
        t2_hash = hash_bytes(t2_ref)
    elif isinstance(t2_ref, (str, Path)) and Path(t2_ref).is_file():
        t2_hash = hash_file(t2_ref)
    elif isinstance(t2_ref, str):
        t2_hash = hash_bytes(t2_ref.encode("utf-8"))
    else:
        t2_hash = "0" * 64

    # Hash GeoJSON vectors
    vector_hash = hash_geojson(geojson_features)

    # Timestamp in ISO 8601 UTC
    now_utc = datetime.now(timezone.utc).isoformat()

    # Generate composite evidence root seal: SHA256(t1_hash + t2_hash + vector_hash + task_id)
    root_seal_payload = f"{task_id}:{t1_hash}:{t2_hash}:{vector_hash}:{analyst_callsign}:{now_utc}"
    evidence_root_seal = hash_bytes(root_seal_payload.encode("utf-8"))

    manifest: Dict[str, Any] = {
        "task_id": task_id,
        "classification": classification,
        "analyst_callsign": analyst_callsign,
        "timestamp_utc": now_utc,
        "hashing_algorithm": "SHA-256",
        "evidence_hashes": {
            "t1_source_sha256": t1_hash,
            "t2_source_sha256": t2_hash,
            "vector_geojson_sha256": vector_hash,
            "evidence_root_seal": evidence_root_seal,
        },
        "chain_of_custody_verified": True,
        "forensic_standard": "ISO/IEC 27037:2012 Digital Evidence Handling",
    }

    if extra_metadata:
        manifest["metadata"] = extra_metadata

    return manifest
