"""
Geospatial Guardrails & Physical Invariant Validators for GeoDelta
Enforces:
1. Law 1: Nyquist-Shannon GSD Resolution Invariant (D >= 2 * GSD)
2. Law 4: Optical Cloud Occlusion Threshold (Cloud Cover <= 35%)
"""

import re
from typing import Tuple, Optional, Dict
import numpy as np

from app.core.exceptions import SubNyquistResolutionException, CloudCoverExceededException
from app.core.config import settings

# Catalog of characteristic dimensions (meters) for tactical and civilian entities
TACTICAL_ENTITY_CATALOG: Dict[str, float] = {
    # Sub-Nyquist entities on 10m GSD (dimension < 20m)
    "individual vehicle": 4.5,
    "vehicle": 4.5,
    "car": 4.0,
    "truck": 7.0,
    "tank": 7.5,
    "armored vehicle": 7.0,
    "apc": 6.5,
    "artillery piece": 6.0,
    "artillery": 6.0,
    "single tree": 3.5,
    "tree": 3.5,
    "tent": 5.0,
    "small boat": 6.0,
    "boat": 6.0,
    "gun position": 8.0,
    "guard post": 4.0,
    "checkpoint": 10.0,
    "sniper hide": 3.0,

    # Resolvable entities on 10m GSD (dimension >= 20m)
    "airstrip": 80.0,
    "runway": 60.0,
    "paved runway": 60.0,
    "runway extension": 50.0,
    "perimeter revetment": 30.0,
    "revetment": 30.0,
    "fortification": 40.0,
    "bunker complex": 35.0,
    "barracks": 45.0,
    "hangar": 50.0,
    "forward operating base": 150.0,
    "fob": 150.0,
    "helipad": 25.0,
    "road grading": 25.0,
    "roadway": 20.0,
    "unpaved road": 20.0,
    "trench network": 40.0,
    "ammunition depot": 60.0,
    "clearings": 50.0,
    "agricultural clearing": 100.0,
}


def validate_nyquist_resolution(query_text: str, native_gsd: float = 10.0) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Law 1: The Nyquist-Shannon GSD Resolution Invariant
    Validates whether any target entity mentioned in the query can be physically
    resolved under the Nyquist-Shannon sampling theorem (D >= 2 * GSD).
    
    If violated, raises SubNyquistResolutionException.
    """
    normalized_query = query_text.lower()
    min_resolvable_dimension = 2.0 * native_gsd

    # Sort catalog entities by length descending to match most specific entity first
    sorted_entities = sorted(TACTICAL_ENTITY_CATALOG.items(), key=lambda item: len(item[0]), reverse=True)

    for entity_name, dimension in sorted_entities:
        # Match entity as a distinct word boundary
        pattern = r"\b" + re.escape(entity_name) + r"\b"
        if re.search(pattern, normalized_query):
            if dimension < min_resolvable_dimension:
                raise SubNyquistResolutionException(
                    entity_name=entity_name,
                    entity_dimension=dimension,
                    native_gsd=native_gsd
                )
            return True, entity_name, dimension

    # If no specific catalog entity is matched, query passes feasibility screening
    return True, None, None


def validate_cloud_cover(
    cloud_mask: np.ndarray,
    max_threshold: Optional[float] = None
) -> float:
    """
    Law 4: Optical Cloud Occlusion Guardrail (NFR-SAFE-001)
    Calculates the ratio of cloud/cirrus pixels within the selected AOI bounding box.
    If ratio exceeds max_threshold (default: 35%), trips CloudCoverExceededException.
    
    Returns the computed cloud fraction in [0.0, 1.0].
    """
    threshold = max_threshold if max_threshold is not None else settings.MAX_PERMITTED_CLOUD_RATIO

    if cloud_mask.size == 0:
        return 0.0

    cloud_pixels = np.count_nonzero(cloud_mask)
    cloud_ratio = float(cloud_pixels) / float(cloud_mask.size)

    if cloud_ratio > threshold:
        raise CloudCoverExceededException(cloud_ratio=cloud_ratio, threshold=threshold)

    return cloud_ratio
