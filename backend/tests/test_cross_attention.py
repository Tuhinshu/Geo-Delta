"""
Unit tests for Dual-Stream Siamese Cross-Attention Engine & UNet++ Decoder (FR-INF-001, FR-INF-002).
"""

import pytest
import torch
import torch.nn.functional as F

from app.services.siamese_engine import SiameseCrossAttentionEngine


def test_siamese_cross_attention_intermediate_shapes():
    """
    FR-INF-001: Validates tensor shapes across all intermediate stages for 256x256 input.
    """
    model = SiameseCrossAttentionEngine(in_channels=4, embed_dim=512)
    model.eval()

    b, c, h, w = 1, 4, 256, 256
    i_t1 = torch.rand(b, c, h, w, dtype=torch.float32)
    i_t2 = torch.rand(b, c, h, w, dtype=torch.float32)
    e_star = F.normalize(torch.randn(1, 512), p=2, dim=-1)

    with torch.no_grad():
        f_t1_pyramid, f_t2_pyramid, f_delta_pyramid = model.extract_feature_pyramids(i_t1, i_t2)

        # 1. Stage 1: C=64, H/4, W/4 -> (1, 64, 64, 64)
        assert f_t1_pyramid[0].shape == (b, 64, h // 4, w // 4)
        assert f_t2_pyramid[0].shape == (b, 64, h // 4, w // 4)
        assert f_delta_pyramid[0].shape == (b, 64, h // 4, w // 4)

        # 2. Stage 2: C=128, H/8, W/8 -> (1, 128, 32, 32)
        assert f_t1_pyramid[1].shape == (b, 128, h // 8, w // 8)
        assert f_delta_pyramid[1].shape == (b, 128, h // 8, w // 8)

        # 3. Stage 3: C=256, H/16, W/16 -> (1, 256, 16, 16)
        assert f_t1_pyramid[2].shape == (b, 256, h // 16, w // 16)
        assert f_delta_pyramid[2].shape == (b, 256, h // 16, w // 16)

        # 4. Stage 4 Bottleneck: C=512, H/32, W/32 -> (1, 512, 8, 8)
        assert f_t1_pyramid[3].shape == (b, 512, h // 32, w // 32)
        assert f_delta_pyramid[3].shape == (b, 512, h // 32, w // 32)

        # 5. Cross-attention modulation at bottleneck
        f_fused_4 = model.cross_attention(f_delta_pyramid[3], e_star)
        assert f_fused_4.shape == (b, 512, h // 32, w // 32)

        # 6. UNet++ decoder full restoration -> (1, 1, 256, 256)
        prob_map = model.forward(i_t1, i_t2, e_star)
        assert prob_map.shape == (b, 1, h, w)
        assert prob_map.min().item() >= 0.0
        assert prob_map.max().item() <= 1.0


def test_siamese_fast_counterfactual_forward():
    """
    FR-NLQ-004: Tests forward_from_features bypassing ResNet-34 encoders.
    """
    model = SiameseCrossAttentionEngine(in_channels=4, embed_dim=512)
    model.eval()

    b, c, h, w = 1, 4, 128, 128
    i_t1 = torch.rand(b, c, h, w)
    i_t2 = torch.rand(b, c, h, w)
    e1 = F.normalize(torch.randn(1, 512), p=2, dim=-1)
    e2 = F.normalize(torch.randn(1, 512), p=2, dim=-1)

    with torch.no_grad():
        # Pre-extract pyramids
        _, _, f_delta_pyramid = model.extract_feature_pyramids(i_t1, i_t2)

        # Fast forward pass for prompt 1
        prob1 = model.forward_from_features(f_delta_pyramid, e1, target_size=(h, w))
        # Fast forward pass for prompt 2 (counter-factual)
        prob2 = model.forward_from_features(f_delta_pyramid, e2, target_size=(h, w))

        assert prob1.shape == (b, 1, h, w)
        assert prob2.shape == (b, 1, h, w)
        # Differing prompts produce differing conditioned activation patterns
        assert not torch.equal(prob1, prob2)
        assert (prob1 - prob2).abs().max().item() > 0.0

        # Bottleneck features exhibit significant prompt modulation
        f_fused_1 = model.cross_attention(f_delta_pyramid[3], e1)
        f_fused_2 = model.cross_attention(f_delta_pyramid[3], e2)
        assert (f_fused_1 - f_fused_2).abs().max().item() > 0.5
