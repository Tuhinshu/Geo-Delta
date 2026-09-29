"""
Unit tests for Geospatial Guardrails & Physical Invariants (Law 1 & Law 4).
"""

import pytest
import numpy as np

from app.core.guardrails import validate_nyquist_resolution, validate_cloud_cover
from app.core.exceptions import SubNyquistResolutionException, CloudCoverExceededException


def test_nyquist_sub_resolution_rejections():
    """
    Law 1: Target entities with characteristic dimension D < 2 * GSD must be rejected.
    For Sentinel-2 (10m GSD), 2 * GSD = 20m.
    """
    # 4.5m vehicle < 20m -> should raise SubNyquistResolutionException
    with pytest.raises(SubNyquistResolutionException) as exc_info:
        validate_nyquist_resolution("Detect individual vehicle crossing road", native_gsd=10.0)
    assert exc_info.value.details["entity"] == "vehicle" or exc_info.value.details["entity"] == "individual vehicle"
    assert exc_info.value.details["native_gsd"] == 10.0

    # 3.5m tree < 20m
    with pytest.raises(SubNyquistResolutionException):
        validate_nyquist_resolution("Show single tree clearing", native_gsd=10.0)

    # 7.5m tank < 20m
    with pytest.raises(SubNyquistResolutionException):
        validate_nyquist_resolution("Identify tank movements near border", native_gsd=10.0)


def test_nyquist_resolvable_entities():
    """
    Law 1: Target entities with characteristic dimension D >= 2 * GSD must pass.
    """
    # 60m runway extension >= 20m -> passes
    feasible, entity, dim = validate_nyquist_resolution("Identify runway extension", native_gsd=10.0)
    assert feasible is True
    assert entity == "runway extension"
    assert dim == 50.0 or dim == 60.0

    # 30m perimeter revetment >= 20m -> passes
    feasible, entity, dim = validate_nyquist_resolution("Inspect perimeter revetment", native_gsd=10.0)
    assert feasible is True
    assert entity == "perimeter revetment"

    # Query without specific catalog keywords -> passes general screening
    feasible, entity, dim = validate_nyquist_resolution("Show area change between winter and spring", native_gsd=10.0)
    assert feasible is True
    assert entity is None


def test_nyquist_high_resolution_sensor():
    """
    Under high-resolution sensors (e.g. Cartosat-2S at 0.5m GSD),
    minimum resolvable dimension is 1.0m, so vehicles (4.5m) must pass.
    """
    feasible, entity, dim = validate_nyquist_resolution("Show vehicle in sector", native_gsd=0.5)
    assert feasible is True
    assert entity == "vehicle"
    assert dim == 4.5


def test_cloud_cover_guardrail():
    """
    Law 4 & NFR-SAFE-001: Cloud cover ratio > 35% must trip CloudCoverExceededException.
    """
    # 40% cloud cover -> Exceeds 35% threshold
    mask_40 = np.zeros((100, 100), dtype=np.uint8)
    mask_40[:40, :] = 1  # 4000 / 10000 = 40%
    with pytest.raises(CloudCoverExceededException) as exc_info:
        validate_cloud_cover(mask_40, max_threshold=0.35)
    assert exc_info.value.details["cloud_ratio"] == 0.40

    # 15% cloud cover -> Within acceptable limits
    mask_15 = np.zeros((100, 100), dtype=np.uint8)
    mask_15[:15, :] = 1  # 1500 / 10000 = 15%
    ratio = validate_cloud_cover(mask_15, max_threshold=0.35)
    assert ratio == 0.15

    # 0% cloud cover
    mask_0 = np.zeros((50, 50), dtype=np.uint8)
    assert validate_cloud_cover(mask_0, max_threshold=0.35) == 0.0

    # Empty mask
    empty_mask = np.zeros((0, 0), dtype=np.uint8)
    assert validate_cloud_cover(empty_mask, max_threshold=0.35) == 0.0
