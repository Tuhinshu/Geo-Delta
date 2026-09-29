"""
Unit tests for RemoteCLIP Text Encoder & Negative Suppression Engine (FR-NLQ-002, FR-NLQ-003).
"""

import pytest
import torch
import torch.nn.functional as F

from app.services.vlm_encoder import RemoteCLIPTextEncoder


def test_vlm_encoder_single_prompt_projection():
    """
    FR-NLQ-002: Prompt must be projected into a 512-dimensional normalized unit vector.
    """
    encoder = RemoteCLIPTextEncoder(embed_dim=512)
    prompt = "Show newly paved runway extension"

    embedding = encoder.encode_text(prompt)

    # 1. Shape validation: (1, 512)
    assert embedding.shape == (1, 512)
    assert embedding.dtype == torch.float32

    # 2. L2 Unit Normalization validation: ||e_t||_2 == 1.0
    l2_norm = torch.norm(embedding, p=2, dim=-1).item()
    assert abs(l2_norm - 1.0) < 1e-5


def test_vlm_encoder_negative_suppression_vector():
    """
    FR-NLQ-003: Negative semantic suppression vector:
        e* = (e_t - beta * e_neg) / ||e_t - beta * e_neg||_2
    Cancels directional projection corresponding to agricultural / seasonal noise.
    """
    encoder = RemoteCLIPTextEncoder(embed_dim=512, default_beta=0.65)
    prompt = "Identify fortified perimeter revetments"
    negative_prompt = "seasonal agricultural harvesting and crop clearings"

    e_t = encoder.encode_text(prompt)
    e_neg = encoder.encode_text(negative_prompt)
    e_star = encoder.encode_conditioned_prompt(prompt, negative_query=negative_prompt, beta=0.65)

    # 1. Output shape & unit norm
    assert e_star.shape == (1, 512)
    l2_norm = torch.norm(e_star, p=2, dim=-1).item()
    assert abs(l2_norm - 1.0) < 1e-5

    # 2. Mathematical suppression validation:
    # Cosine similarity between e* and e_neg must be lower than between e_t and e_neg
    sim_t_neg = F.cosine_similarity(e_t, e_neg).item()
    sim_star_neg = F.cosine_similarity(e_star, e_neg).item()

    assert sim_star_neg < sim_t_neg


def test_vlm_encoder_empty_negative_prompt():
    """
    When negative prompt is None or empty string, e* must exactly equal e_t.
    """
    encoder = RemoteCLIPTextEncoder(embed_dim=512)
    prompt = "Newly graded unpaved roadway"

    e_t = encoder.encode_text(prompt)
    e_star = encoder.encode_conditioned_prompt(prompt, negative_query=None)

    assert torch.allclose(e_t, e_star, atol=1e-5)

    e_star_empty = encoder.encode_conditioned_prompt(prompt, negative_query="   ")
    assert torch.allclose(e_t, e_star_empty, atol=1e-5)
