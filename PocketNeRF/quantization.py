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


class LearnedBitwidthQuantizer(nn.Module):
    """
    Quantizer with learnable bitwidth (soft bitwidth from A-CAQ paper).
    """
    def __init__(self, init_bits=8.0, min_bits=2.0, max_bits=32.0, symmetric=True):
        super(LearnedBitwidthQuantizer, self).__init__()
        
        # Soft bitwidth parameter
        self.soft_bits = nn.Parameter(torch.tensor(float(init_bits)))
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.symmetric = symmetric
        
        # Range scale and offset (learnable)
        self.range_scale = nn.Parameter(torch.tensor(1.0))
        if not symmetric:
            self.v_max = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('v_max', None)
    
    @property
    def bit_width(self):
        """Get the current (soft) bitwidth."""
        return torch.clamp(self.soft_bits, self.min_bits, self.max_bits)
    
    @property
    def integer_bit_width(self):
        """Get the rounded integer bitwidth."""
        return torch.round(self.bit_width)
    
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
        if not self.training:
            # Use integer bitwidth during inference
            B = int(self.integer_bit_width.item())
        else:
            # Use soft bitwidth during training for gradients
            B = self.bit_width
            
        qmin, qmax = self.get_quantization_params()
        
        # Calculate step size
        if self.symmetric:
            # For symmetric quantization
            scale = self.range_scale
            zero_point = 0
        else:
            # For asymmetric quantization
            scale = self.range_scale / (2 ** B - 1)
            zero_point = torch.round(-self.v_max / scale)
        
        if self.training:
            # Simulate quantization with straight-through estimator
            x_scaled = x / scale
            x_quant = torch.round(x_scaled + zero_point)
            x_quant = torch.clamp(x_quant, qmin, qmax)
            x_dequant = (x_quant - zero_point) * scale
            
            # Straight-through estimator
            return x + (x_dequant - x).detach()
        else:
            # Actual quantization for inference
            x_scaled = x / scale
            x_quant = torch.round(x_scaled + zero_point)
            x_quant = torch.clamp(x_quant, qmin, qmax)
            return (x_quant - zero_point) * scale
    
    def extra_repr(self):
        return (f'soft_bits={self.soft_bits.data:.2f}, '
                f'range=[{self.min_bits}, {self.max_bits}], '
                f'symmetric={self.symmetric}')


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