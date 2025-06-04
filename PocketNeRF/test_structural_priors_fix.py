"""
Test script to verify structural priors fixes work correctly.
"""

import torch
import numpy as np
import sys
import os

# Add the current directory to the path
sys.path.append('.')

from run_nerf_helpers import NeRFSmall
from structural_priors_v2 import combine_structural_losses_v2

def test_network_output_shapes():
    """Test that NeRFSmall outputs correct shapes with and without normal prediction."""
    print("Testing NeRF network output shapes...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test without normal prediction
    model_no_normals = NeRFSmall(
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        input_ch=63,  # Hash embedding output
        input_ch_views=16,  # SH encoding output
        predict_normals=False
    ).to(device)
    
    # Test with normal prediction
    model_with_normals = NeRFSmall(
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        input_ch=63,  # Hash embedding output
        input_ch_views=16,  # SH encoding output  
        predict_normals=True
    ).to(device)
    
    # Create test input
    batch_size = 1024
    N_samples = 64
    test_input = torch.randn(batch_size, N_samples, 63 + 16, device=device)
    
    # Test without normals
    with torch.no_grad():
        output_no_normals = model_no_normals(test_input)
        print(f"   Without normals: input {test_input.shape} → output {output_no_normals.shape}")
        assert output_no_normals.shape[-1] == 4, f"Expected 4 channels, got {output_no_normals.shape[-1]}"
    
    # Test with normals
    with torch.no_grad():
        output_with_normals = model_with_normals(test_input)
        print(f"   With normals: input {test_input.shape} → output {output_with_normals.shape}")
        assert output_with_normals.shape[-1] == 7, f"Expected 7 channels, got {output_with_normals.shape[-1]}"
    
    print("   ✅ Network output shapes are correct!")
    return True


def test_structural_priors_error_handling():
    """Test that structural priors handle invalid inputs gracefully."""
    print("Testing structural priors error handling...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with None normals
    depth_pred = torch.randn(100, device=device)
    normals_none = None
    rays_d = torch.randn(100, 3, device=device)
    
    loss, loss_dict = combine_structural_losses_v2(
        depth_pred=depth_pred,
        normals=normals_none,
        rays_d=rays_d
    )
    
    print(f"   None normals: loss={loss.item():.6f}, error='{loss_dict.get('error', 'none')}'")
    assert 'error' in loss_dict and 'normals_none' in loss_dict['error']
    
    # Test with empty normals
    normals_empty = torch.empty(0, 3, device=device)
    loss, loss_dict = combine_structural_losses_v2(
        depth_pred=depth_pred,
        normals=normals_empty,
        rays_d=rays_d
    )
    
    print(f"   Empty normals: loss={loss.item():.6f}, error='{loss_dict.get('error', 'none')}'")
    assert 'error' in loss_dict and 'empty_normals' in loss_dict['error']
    
    # Test with invalid shape normals
    normals_invalid = torch.randn(100, 5, device=device)  # Wrong shape
    loss, loss_dict = combine_structural_losses_v2(
        depth_pred=depth_pred,
        normals=normals_invalid,
        rays_d=rays_d
    )
    
    print(f"   Invalid shape normals: loss={loss.item():.6f}, error='{loss_dict.get('error', 'none')}'")
    assert 'error' in loss_dict and 'invalid_normals_shape' in loss_dict['error']
    
    # Test with size mismatch
    normals_mismatch = torch.randn(50, 3, device=device)  # Different size than depth
    loss, loss_dict = combine_structural_losses_v2(
        depth_pred=depth_pred,
        normals=normals_mismatch,
        rays_d=rays_d
    )
    
    print(f"   Size mismatch: loss={loss.item():.6f}, error='{loss_dict.get('error', 'none')}'")
    assert 'error' in loss_dict and 'size_mismatch' in loss_dict['error']
    
    print("   ✅ Error handling is working correctly!")
    return True


def test_structural_priors_valid_input():
    """Test that structural priors work with valid inputs."""
    print("Testing structural priors with valid inputs...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create valid test data
    N_rays = 1000
    depth_pred = torch.abs(torch.randn(N_rays, device=device)) + 0.1  # Positive depths
    normals = torch.randn(N_rays, 3, device=device)
    normals = torch.nn.functional.normalize(normals, dim=-1)  # Unit normals
    rays_d = torch.randn(N_rays, 3, device=device)
    spatial_coords = torch.rand(N_rays, 2, device=device) * 100  # Random 2D coordinates
    
    # Test with valid inputs
    weights = {
        'manhattan': 0.001,
        'planarity': 0.002,
        'normal_consistency': 0.0005
    }
    
    loss, loss_dict = combine_structural_losses_v2(
        depth_pred=depth_pred,
        normals=normals,
        rays_d=rays_d,
        spatial_coords=spatial_coords,
        weights=weights
    )
    
    print(f"   Valid inputs: loss={loss.item():.6f}")
    print(f"   Loss components: {list(loss_dict.keys())}")
    
    # Check that we got meaningful loss values
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.item() >= 0, "Loss should be non-negative"
    assert 'error' not in loss_dict, "Should not have errors with valid input"
    
    # Check semantic detection
    floor_count = loss_dict.get('semantic_floor_count', 0)
    wall_count = loss_dict.get('semantic_wall_count', 0)
    print(f"   Detected semantics: {floor_count} floor, {wall_count} wall points")
    
    print("   ✅ Structural priors working with valid inputs!")
    return True


def main():
    """Run all tests."""
    print("🧪 Testing PocketNeRF Structural Priors Fixes")
    print("=" * 50)
    
    try:
        test_network_output_shapes()
        print()
        test_structural_priors_error_handling()
        print()
        test_structural_priors_valid_input()
        print()
        print("🎉 All tests passed! The fixes should work correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main() 