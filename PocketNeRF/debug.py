import torch
import torch.nn as nn
from quantization import LearnedBitwidthQuantizer, FakeQuantizer

# Test the quantizer behavior
def test_quantizer():
    # Create a test quantizer
    quantizer = LearnedBitwidthQuantizer(
        init_bits=32.0,
        min_bits=2.0,
        max_bits=32.0,
        symmetric=False
    )
    
    # Test input
    x = torch.randn(10, 10) * 0.1  # Small values like hash embeddings
    
    print("Input stats:")
    print(f"  Mean: {x.mean().item():.6f}")
    print(f"  Std: {x.std().item():.6f}")
    print(f"  Min: {x.min().item():.6f}")
    print(f"  Max: {x.max().item():.6f}")
    
    # Apply quantization
    with torch.no_grad():
        x_quant = quantizer(x)
    
    print("\nOutput stats:")
    print(f"  Mean: {x_quant.mean().item():.6f}")
    print(f"  Std: {x_quant.std().item():.6f}")
    print(f"  Min: {x_quant.min().item():.6f}")
    print(f"  Max: {x_quant.max().item():.6f}")
    
    print("\nQuantizer params:")
    print(f"  Soft bits: {quantizer.soft_bits.item():.2f}")
    print(f"  Integer bits: {quantizer.integer_bit_width}")
    print(f"  Range scale: {quantizer.range_scale.item():.6f}")
    if hasattr(quantizer, 'v_max'):
        print(f"  V_max: {quantizer.v_max.item():.6f}")
    
    print("\nError:")
    error = (x - x_quant).abs()
    print(f"  Mean absolute error: {error.mean().item():.6f}")
    print(f"  Max absolute error: {error.max().item():.6f}")
    
    # Check if quantization is actually happening
    unique_vals = torch.unique(x_quant).numel()
    print(f"\nUnique values in output: {unique_vals}")
    print(f"Expected for {quantizer.integer_bit_width}-bit: {2**quantizer.integer_bit_width}")

import torch
import sys
sys.path.append('PocketNeRF')

# Import both quantizers to compare
from quantization import LearnedBitwidthQuantizer as OldQuantizer

# Test with the fixed quantizer
def test_fixed_quantizer():
    # Simulate hash embedding values
    x = torch.randn(100, 32) * 0.0001  # Small values like real hash embeddings
    
    print("Testing with hash embedding-like values")
    print(f"Input stats: mean={x.mean():.6f}, std={x.std():.6f}, min={x.min():.6f}, max={x.max():.6f}")
    
    # Test old quantizer
    print("\n=== OLD QUANTIZER ===")
    old_q = OldQuantizer(init_bits=8.0, symmetric=False)
    old_q.eval()  # Put in eval mode
    with torch.no_grad():
        x_old = old_q(x)
    print(f"Output stats: mean={x_old.mean():.6f}, std={x_old.std():.6f}")
    print(f"Unique values: {torch.unique(x_old).numel()}")
    print(f"Error: {(x - x_old).abs().mean():.6f}")
    
    # Test fixed quantizer (from the artifact above)
    print("\n=== FIXED QUANTIZER ===")
    # You'll need to copy the fixed code and test it
    # This shows what the output should look like
    print("Expected behavior:")
    print("- Output should preserve the scale of input")
    print("- Should have many unique values for 8-bit quantization")
    print("- Error should be small (< 0.0001)")
    
    # Test passthrough for comparison
    print("\n=== PASSTHROUGH (No Quantization) ===")
    print(f"Output would be identical to input")
    print(f"This is what PSNR ~23 dB corresponds to")

def test_high_bit_quantizer():
    """Test quantizer with high bit counts like in training"""
    
    # Test values similar to hash embeddings
    x = torch.randn(100, 32) * 0.0001
    
    print("Input stats:")
    print(f"  Mean: {x.mean():.6f}, Std: {x.std():.6f}")
    print(f"  Min: {x.min():.6f}, Max: {x.max():.6f}")
    
    # Test with different bit widths
    for bits in [8, 16, 32]:
        print(f"\n=== Testing {bits}-bit quantization ===")
        
        quantizer = LearnedBitwidthQuantizer(
            init_bits=float(bits),
            min_bits=2.0,
            max_bits=32.0,
            symmetric=False
        )
        
        # Put in eval mode to avoid training behavior
        quantizer.eval()
        
        with torch.no_grad():
            x_quant = quantizer(x)
        
        print(f"Quantizer params:")
        print(f"  Soft bits: {quantizer.soft_bits.item():.1f}")
        print(f"  Range scale: {quantizer.range_scale.item():.6f}")
        if hasattr(quantizer, 'v_max') and quantizer.v_max is not None:
            print(f"  V_max: {quantizer.v_max.item():.6f}")
        
        print(f"Output stats:")
        print(f"  Mean: {x_quant.mean():.6f}, Std: {x_quant.std():.6f}")
        print(f"  Min: {x_quant.min():.6f}, Max: {x_quant.max():.6f}")
        
        error = (x - x_quant).abs()
        print(f"Error:")
        print(f"  Mean: {error.mean():.6f}, Max: {error.max():.6f}")
        
        unique = torch.unique(x_quant).numel()
        print(f"Unique values: {unique}")
        
        # Check if it's actually quantizing or just passing through
        if unique > 1000:
            print("  WARNING: Too many unique values - might not be quantizing properly!")


if __name__ == "__main__":
    test_high_bit_quantizer()
    # test_fixed_quantizer()
    # test_quantizer()