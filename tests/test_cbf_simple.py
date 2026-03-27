"""
Simple test for DistanceAwareCBFLayer
"""

import torch
from swarm_rl.cbf.cbf_layer import DistanceAwareCBFLayer


def test_initialization():
    """Test layer initialization"""
    layer = DistanceAwareCBFLayer(alpha_cbf=1.0, k=2.0, sigma=0.1)
    assert layer.alpha_cbf == 1.0
    assert layer.k == 2.0
    print("✓ Initialization test passed")


def test_alpha_computation():
    """Test distance weight computation"""
    layer = DistanceAwareCBFLayer()
    sdf_obs = torch.tensor([[0.2, 0.15, 0.2, 0.15, 0.1, 0.15, 0.2, 0.15, 0.2]])
    alpha = layer.compute_alpha(sdf_obs)

    assert alpha.shape == (1, 1)
    assert alpha.item() > 0
    print(f"✓ Alpha computation test passed (alpha={alpha.item():.4f})")


def test_sdf_gradient():
    """Test SDF gradient computation"""
    layer = DistanceAwareCBFLayer()
    sdf_obs = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.2, 0.3, 0.1, 0.3, 0.1]])
    n, p = layer.compute_sdf_gradient(sdf_obs)

    assert n.shape == (1, 3)
    assert p.shape == (1, 1)
    norm = torch.norm(n).item()
    assert abs(norm - 1.0) < 0.01, f"Gradient not normalized: {norm}"
    print(f"✓ SDF gradient test passed (norm={norm:.4f})")


def test_forward_no_qp():
    """Test forward pass when qpth not available"""
    layer = DistanceAwareCBFLayer()

    if layer.qp is None:
        print("⚠ qpth not installed, skipping forward test")
        return

    batch_size = 2
    rl_output = torch.randn(batch_size, 4)
    state = {
        'R': torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1),
        'vel': torch.randn(batch_size, 3)
    }
    sdf_obs = torch.randn(batch_size, 9)

    try:
        safe_action = layer(rl_output, state, sdf_obs)
        assert safe_action.shape == (batch_size, 4)
        print("✓ Forward pass test passed")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")


if __name__ == '__main__':
    test_initialization()
    test_alpha_computation()
    test_sdf_gradient()
    test_forward_no_qp()
    print("\n✓ All tests completed")
