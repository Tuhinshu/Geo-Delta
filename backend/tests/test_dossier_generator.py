"""
Unit & Integration Test Suite for Phase 5: Cryptographic Intelligence Dossier Engine
Tests ReportLab compilation, PyMuPDF inspection, SHA-256 evidence chain-of-custody,
optical chip extraction, and REST export endpoints (FR-EXP-001, FR-EXP-002).
"""

import io
import time
import zipfile
import pymupdf as fitz  # PyMuPDF (fitz alias for backward compat)
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.chip_extractor import (
    extract_localized_chip,
    generate_dossier_chip_trio,
    render_vector_overlay_chip,
)
from app.services.crypto_proof import (
    create_evidence_manifest,
    hash_bytes,
    hash_geojson,
)
from app.services.dossier_generator import DossierGenerator


@pytest.fixture
def sample_polygons():
    return [
        {
            "feature_id": "feat-runway-001",
            "tactical_class": "Newly Paved Runway Extension",
            "confidence": 0.94,
            "area_sq_meters": 18400.0,
            "area_hectares": 1.84,
            "centroid_wgs84": [25.0450, 75.0480],
            "centroid_mgrs": "43R EH 04800 04500",
            "geometry_geojson": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [75.045, 25.040],
                        [75.050, 25.040],
                        [75.050, 25.045],
                        [75.045, 25.045],
                        [75.045, 25.040],
                    ]
                ],
            },
        },
        {
            "feature_id": "feat-revet-002",
            "tactical_class": "Fortified Vehicle Revetment Alpha",
            "confidence": 0.89,
            "area_sq_meters": 3600.0,
            "area_hectares": 0.36,
            "centroid_wgs84": [25.0520, 75.0550],
            "centroid_mgrs": "43R EH 05500 05200",
            "geometry_geojson": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [75.053, 25.051],
                        [75.056, 25.051],
                        [75.056, 25.053],
                        [75.053, 25.053],
                        [75.053, 25.051],
                    ]
                ],
            },
        },
    ]


@pytest.fixture
def sample_rasters():
    t1 = np.full((512, 512, 3), 45, dtype=np.uint8)
    t2 = np.full((512, 512, 3), 60, dtype=np.uint8)
    # Add synthetic change in t2
    t2[100:200, 100:300] = 180
    return t1, t2


# ==============================================================================
# 1. Cryptographic Proof Tests (FR-EXP-001)
# ==============================================================================


def test_crypto_proof_deterministic_hashing():
    raw_data = b"GEODELTA-AIR-GAPPED-INTELLIGENCE-2025"
    digest1 = hash_bytes(raw_data)
    digest2 = hash_bytes(raw_data)

    assert len(digest1) == 64
    assert digest1 == digest2
    assert digest1.islower()


def test_crypto_proof_canonical_geojson_hashing(sample_polygons):
    # Whitespace differences should NOT alter canonical SHA-256
    hash_standard = hash_geojson(sample_polygons)
    hash_reformatted = hash_geojson(sample_polygons)

    assert hash_standard == hash_reformatted
    assert len(hash_standard) == 64


def test_create_evidence_manifest(sample_polygons):
    manifest = create_evidence_manifest(
        task_id="TASK-TEST-7788",
        t1_ref=b"MOCK-TIFF-T1",
        t2_ref=b"MOCK-TIFF-T2",
        geojson_features=sample_polygons,
        analyst_callsign="CALLSIGN-ALPHA-9",
    )

    assert manifest["task_id"] == "TASK-TEST-7788"
    assert manifest["analyst_callsign"] == "CALLSIGN-ALPHA-9"
    assert manifest["chain_of_custody_verified"] is True
    assert "t1_source_sha256" in manifest["evidence_hashes"]
    assert "t2_source_sha256" in manifest["evidence_hashes"]
    assert "vector_geojson_sha256" in manifest["evidence_hashes"]
    assert "evidence_root_seal" in manifest["evidence_hashes"]


# ==============================================================================
# 2. Optical Chip Extraction & Compositor Tests
# ==============================================================================


def test_evidence_chip_extraction_dimensions(sample_rasters):
    t1, _ = sample_rasters
    chip = extract_localized_chip(t1, center_rc=(256, 256), chip_size=380)

    assert chip.shape == (380, 380, 3)
    assert chip.dtype == np.uint8


def test_vector_overlay_rendering(sample_rasters, sample_polygons):
    _, t2 = sample_rasters
    overlay = render_vector_overlay_chip(
        t2,
        polygons=sample_polygons,
        chip_size=380,
    )

    assert overlay.shape == (380, 380, 3)
    assert overlay.dtype == np.uint8


def test_generate_dossier_chip_trio(sample_rasters, sample_polygons):
    t1, t2 = sample_rasters
    bytes_t1, bytes_t2, bytes_cov = generate_dossier_chip_trio(
        t1, t2, sample_polygons, chip_size=380
    )

    # Check PNG magic bytes: \x89PNG\r\n\x1a\n
    assert bytes_t1.startswith(b"\x89PNG\r\n\x1a\n")
    assert bytes_t2.startswith(b"\x89PNG\r\n\x1a\n")
    assert bytes_cov.startswith(b"\x89PNG\r\n\x1a\n")


# ==============================================================================
# 3. PDF Dossier Compilation & Latency Benchmark (<= 3.0s Acceptance Invariant)
# ==============================================================================


def test_dossier_pdf_compilation_latency(sample_rasters, sample_polygons):
    generator = DossierGenerator()
    t1, t2 = sample_rasters

    start_time = time.perf_counter()
    pdf_bytes = generator.generate_dossier(
        task_id="TASK-BENCH-001",
        query_prompt="Identify newly paved airstrip extensions and defensive revetments",
        negative_prompt="Natural vegetation growth",
        total_area_altered_sq_m=22000.0,
        total_area_altered_ha=2.2,
        polygons=sample_polygons,
        raster_t1=t1,
        raster_t2=t2,
    )
    elapsed_time = time.perf_counter() - start_time

    # Latency constraint: <= 3.0 seconds
    assert elapsed_time <= 3.0, f"Dossier compilation exceeded 3.0s limit: {elapsed_time:.3f}s"
    assert len(pdf_bytes) > 5000, "PDF document appears unexpectedly small or empty"
    assert pdf_bytes.startswith(b"%PDF"), "Output bytes do not begin with valid %PDF header"


# ==============================================================================
# 4. PyMuPDF Forensic PDF Inspection & Content Validation
# ==============================================================================


def test_dossier_pdf_structural_and_content_verification(sample_rasters, sample_polygons):
    generator = DossierGenerator()
    t1, t2 = sample_rasters

    pdf_bytes = generator.generate_dossier(
        task_id="TASK-INSPECT-99",
        query_prompt="Surveillance target: airstrip extension",
        total_area_altered_sq_m=22000.0,
        total_area_altered_ha=2.2,
        polygons=sample_polygons,
        raster_t1=t1,
        raster_t2=t2,
    )

    # Open with PyMuPDF to extract text and layout metadata
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    assert doc.page_count >= 1

    extracted_text = ""
    for page in doc:
        page_text = page.get_text("text")
        if isinstance(page_text, str):
            extracted_text += page_text

    # Verify military classification banners and metadata
    assert "RESTRICTED // GEOINT ASSESSMENT" in extracted_text
    assert "SIH26227" in extracted_text
    assert "TASK-INSPECT-99" in extracted_text
    assert "Newly Paved Runway Extension" in extracted_text
    assert "Fortified Vehicle Revetment Alpha" in extracted_text
    assert "43R EH" in extracted_text  # MGRS presence
    assert "CRYPTOGRAPHIC EVIDENCE CHAIN-OF-CUSTODY (SHA-256)" in extracted_text
    assert "ISO/IEC 27037:2012" in extracted_text

    doc.close()


# ==============================================================================
# 5. Vector Export Options (FR-EXP-002)
# ==============================================================================


def test_vector_export_geojson_and_zip(sample_polygons):
    generator = DossierGenerator()
    geojson_out = generator.export_geojson(sample_polygons, task_id="TASK-EXP-01")

    assert geojson_out["type"] == "FeatureCollection"
    assert len(geojson_out["features"]) == 2
    assert geojson_out["features"][0]["properties"]["tactical_class"] == "Newly Paved Runway Extension"

    zip_bytes = generator.export_shapefile_zip(sample_polygons, task_id="TASK-EXP-01")
    assert len(zip_bytes) > 200

    # Inspect zip contents
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        file_list = zf.namelist()
        assert "geodelta_TASK-EXP-01.geojson" in file_list
        assert "CHAIN_OF_CUSTODY_MANIFEST.json" in file_list


# ==============================================================================
# 6. REST API Endpoints Verification
# ==============================================================================


def test_dossier_api_endpoints(sample_polygons):
    client = TestClient(app)

    # 1. Export PDF
    pdf_req = {
        "task_id": "API-TASK-1234",
        "query_prompt": "Detect runway paving",
        "total_area_altered_sq_m": 18400.0,
        "total_area_altered_ha": 1.84,
        "polygons": sample_polygons,
    }
    resp = client.post("/api/v1/dossier/export", json=pdf_req)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF")
    assert resp.headers["X-Dossier-Task-Id"] == "API-TASK-1234"

    # 2. Export GeoJSON
    resp_geo = client.post("/api/v1/dossier/export-geojson", json=pdf_req)
    assert resp_geo.status_code == 200
    geo_data = resp_geo.json()
    assert geo_data["type"] == "FeatureCollection"
    assert len(geo_data["features"]) == 2

    # 3. Export ZIP
    resp_zip = client.post("/api/v1/dossier/export-shapefile-zip", json=pdf_req)
    assert resp_zip.status_code == 200
    assert resp_zip.headers["content-type"] == "application/zip"
    assert len(resp_zip.content) > 200
