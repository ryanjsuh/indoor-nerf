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

if __name__ == "__main__":
    test_quantizer()