import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FakeQuantizer(nn.Module):
    """
    Fake quantizer that simulates quantization during training.
    Based on the A-CAQ paper's approach.
    """
    def __init__(self, num_bits=8, symmetric=True, initialize_scale=True):
        super(FakeQuantizer, self).__init__()
        self.num_bits = num_bits
        self.symmetric = symmetric
        
        # Quantization range
        if symmetric:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1
            
        # Learnable parameters
        self.scale = nn.Parameter(torch.tensor(1.0))
        if not symmetric:
            self.zero_point = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('zero_point', torch.tensor(0.0))
    
    def forward(self, x):
        if self.training:
            # During training, simulate quantization
            # Scale the input
            x_scaled = x / self.scale
            
            # Add zero point if asymmetric
            if not self.symmetric:
                x_scaled = x_scaled + self.zero_point
            
            # Round to nearest integer (using straight-through estimator)
            x_quant = torch.round(x_scaled)
            
            # Clamp to quantization range
            x_quant = torch.clamp(x_quant, self.qmin, self.qmax)
            
            # Dequantize
            x_dequant = x_quant - self.zero_point
            x_dequant = x_dequant * self.scale
            
            # Straight-through estimator: use dequantized values but 
            # gradient flows through as if no quantization
            return x + (x_dequant - x).detach()
        else:
            # During inference, perform actual quantization
            x_scaled = x / self.scale
            if not self.symmetric:
                x_scaled = x_scaled + self.zero_point
            x_quant = torch.round(x_scaled)
            x_quant = torch.clamp(x_quant, self.qmin, self.qmax)
            x_dequant = (x_quant - self.zero_point) * self.scale
            return x_dequant
    
    def extra_repr(self):
        return f'num_bits={self.num_bits}, symmetric={self.symmetric}'

#Edited version
class LearnedBitwidthQuantizer(nn.Module):
    """
    Quantizer with learnable bitwidth (soft bitwidth from A-CAQ paper).
    Fixed to handle small values properly.
    """
    def __init__(self, init_bits=8.0, min_bits=2.0, max_bits=32.0, symmetric=True):
        super(LearnedBitwidthQuantizer, self).__init__()
        
        # Soft bitwidth parameter
        self.soft_bits = nn.Parameter(torch.tensor(float(init_bits)))
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.symmetric = symmetric
        
        # Initialize with small scales appropriate for hash embeddings
        # Hash embeddings are typically initialized to ~0.0001
        self.range_scale = nn.Parameter(torch.tensor(0.0002))  # Small initial scale
        
        if not symmetric:
            # For asymmetric, v_max should be close to expected max value
            self.v_max = nn.Parameter(torch.tensor(0.0001))
        else:
            self.register_buffer('v_max', None)
        
        # Add calibration flag
        self.calibrated = False
        self.register_buffer('running_min', torch.tensor(float('inf')))
        self.register_buffer('running_max', torch.tensor(float('-inf')))
    
    def calibrate(self, x):
        """Calibrate quantizer parameters based on actual data statistics."""
        with torch.no_grad():
            # Update running min/max
            batch_min = x.min()
            batch_max = x.max()
            
            self.running_min = torch.min(self.running_min, batch_min)
            self.running_max = torch.max(self.running_max, batch_max)
            
            # Update scale based on actual range
            data_range = self.running_max - self.running_min
            
            if self.symmetric:
                # For symmetric quantization, scale based on max absolute value
                max_abs = torch.max(torch.abs(self.running_min), torch.abs(self.running_max))
                self.range_scale.data = 2 * max_abs
            else:
                # For asymmetric quantization
                self.range_scale.data = data_range
                self.v_max.data = self.running_max
            
            self.calibrated = True
    
    @property
    def bit_width(self):
        """Get the current (soft) bitwidth."""
        return torch.clamp(self.soft_bits, self.min_bits, self.max_bits)
    
    @property
    def integer_bit_width(self):
        """Get the rounded integer bitwidth."""
        return int(torch.round(self.bit_width).item())
    
    def get_quantization_params(self):
        """Calculate quantization parameters based on current bitwidth."""
        B = self.integer_bit_width
        
        if self.symmetric:
            qmin = -(2 ** (B - 1))
            qmax = 2 ** (B - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** B - 1
            
        return qmin, qmax
    
    def forward(self, x):
        # Calibrate on first forward pass
        if self.training and not self.calibrated:
            self.calibrate(x)
        
        qmin, qmax = self.get_quantization_params()

        if not self.training:
            B = self.integer_bit_width
        else:
            B = self.bit_width
        
        # Calculate step size with proper scaling
        if self.symmetric:
            # For symmetric quantization
            scale = self.range_scale / (2 ** (B - 1))
            zero_point = 0
        else:
            # For asymmetric quantization
            # Ensure we don't divide by zero
            range_val = torch.clamp(self.range_scale, min=1e-8)
            scale = range_val / (2 ** B - 1)
            
            # Calculate zero point to align with v_max
            if self.v_max is not None:
                zero_point = torch.round(torch.clamp(self.v_max / scale, qmin, qmax))
            else:
                zero_point = 0
        
        if self.training:
            # Simulate quantization with straight-through estimator
            x_scaled = x / (scale + 1e-8)  # Add epsilon to prevent division by zero
            x_quant = torch.round(x_scaled + zero_point)
            x_quant = torch.clamp(x_quant, qmin, qmax)
            x_dequant = (x_quant - zero_point) * scale
            
            # Straight-through estimator
            return x + (x_dequant - x).detach()
        else:
            # Actual quantization for inference
            x_scaled = x / (scale + 1e-8)
            x_quant = torch.round(x_scaled + zero_point)
            x_quant = torch.clamp(x_quant, qmin, qmax)
            return (x_quant - zero_point) * scale
    
    def extra_repr(self):
        return (f'soft_bits={self.soft_bits.data:.2f}, '
                f'range=[{self.min_bits}, {self.max_bits}], '
                f'symmetric={self.symmetric}, '
                f'range_scale={self.range_scale.data:.6f}')


# Also create a simple PassthroughQuantizer for debugging
class PassthroughQuantizer(nn.Module):
    """A quantizer that doesn't quantize - for debugging."""
    def __init__(self, **kwargs):
        super(PassthroughQuantizer, self).__init__()
        self.bit_width = 32.0
        self.integer_bit_width = 32
    
    def forward(self, x):
        return x
    
    def extra_repr(self):
        return 'passthrough'

# Utility function to calculate FQR (Feature Quantization Rate)
def calculate_fqr(quantizers):
    """Calculate average bitwidth across all quantizers."""
    if not quantizers:
        return 32.0  # Default full precision
    
    total_bits = 0
    for q in quantizers:
        if hasattr(q, 'bit_width'):
            total_bits += q.bit_width
        elif hasattr(q, 'num_bits'):
            total_bits += q.num_bits
        else:
            total_bits += 32  # Assume full precision if no bitwidth info
            
    return total_bits / len(quantizers)