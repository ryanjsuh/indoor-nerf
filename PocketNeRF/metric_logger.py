import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import torch
from collections import defaultdict

class MetricsLogger:
    """
    Comprehensive logging system for PocketNeRF experiments.
    Tracks all metrics needed for paper writing and creates visualizations.
    """
    
    def __init__(self, log_dir, experiment_name, config):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.config = config
        
        # Create metrics directory
        self.metrics_dir = os.path.join(log_dir, experiment_name, 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metric storage
        self.metrics = {
            'iteration': [],
            'time': [],
            'loss': [],
            'psnr': [],
            'learning_rate': [],
            'avg_bitwidth': [],
            'bitwidth_distribution': [],
            'component_bitwidths': defaultdict(list),  # Per-component tracking
            'memory_usage': [],
            'inference_time': [],
            'test_psnr': [],
            'test_ssim': [],
            'test_lpips': [],
        }
        
        # Quantization-specific metrics
        self.quant_metrics = {
            'embed_bits': [],
            'mlp_bits': [],
            'activation_bits': [],
            'weight_bits': [],
            'quantization_error': [],
            'bit_operations': [],  # BitOps for computational complexity
            'model_size': [],  # Compressed model size
        }
        
        # A-CAQ specific tracking
        self.acaq_metrics = {
            'target_metric': [],
            'loss_ratio': [],
            'bit_adjustments': [],
            'layer_sensitivity': defaultdict(list),
        }
        
        # Save experiment config
        self.save_config()
        
    def save_config(self):
        """Save experiment configuration for reproducibility"""
        config_path = os.path.join(self.metrics_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=4, default=str)
    
    def log_iteration(self, iteration, time_elapsed, loss, psnr, lr, quantizers=None):
        """Log metrics for current iteration"""
        self.metrics['iteration'].append(iteration)
        self.metrics['time'].append(time_elapsed)
        self.metrics['loss'].append(loss)
        self.metrics['psnr'].append(psnr)
        self.metrics['learning_rate'].append(lr)
        
        # Log quantization metrics if applicable
        if quantizers:
            self.log_quantization_metrics(quantizers)
    
    def log_quantization_metrics(self, quantizers):
        """Log detailed quantization metrics"""
        if not quantizers:
            return
            
        # Collect bitwidths
        all_bits = []
        embed_bits = []
        mlp_bits = []
        
        for name, q_list in quantizers.items():
            for idx, q in enumerate(q_list):
                if hasattr(q, 'soft_bits'):
                    bits = q.soft_bits.item()
                    all_bits.append(bits)
                    
                    # Categorize by component type
                    if 'embed' in name:
                        embed_bits.append(bits)
                        self.metrics['component_bitwidths'][f'{name}_{idx}'].append(bits)
                    elif 'mlp' in name or 'network' in name:
                        mlp_bits.append(bits)
                        self.metrics['component_bitwidths'][f'{name}_{idx}'].append(bits)
        
        # Calculate statistics
        if all_bits:
            self.metrics['avg_bitwidth'].append(np.mean(all_bits))
            self.metrics['bitwidth_distribution'].append(all_bits.copy())
            
            if embed_bits:
                self.quant_metrics['embed_bits'].append(np.mean(embed_bits))
            if mlp_bits:
                self.quant_metrics['mlp_bits'].append(np.mean(mlp_bits))
    
    def log_test_metrics(self, iteration, psnr, ssim=None, lpips=None):
        """Log test set evaluation metrics"""
        self.metrics['test_psnr'].append((iteration, psnr))
        if ssim is not None:
            self.metrics['test_ssim'].append((iteration, ssim))
        if lpips is not None:
            self.metrics['test_lpips'].append((iteration, lpips))
    
    def log_acaq_update(self, target_metric, loss_ratio, bit_adjustments):
        """Log A-CAQ specific metrics"""
        self.acaq_metrics['target_metric'].append(target_metric)
        self.acaq_metrics['loss_ratio'].append(loss_ratio)
        self.acaq_metrics['bit_adjustments'].append(bit_adjustments)
    
    def calculate_model_complexity(self, model, quantizers):
        """Calculate BitOps and model size for complexity analysis"""
        total_bitops = 0
        total_bits = 0
        
        # Calculate for each layer
        for name, param in model.named_parameters():
            param_bits = 32  # Default full precision
            
            # Find corresponding quantizer
            for q_name, q_list in quantizers.items():
                if name in q_name:
                    if hasattr(q_list[0], 'soft_bits'):
                        param_bits = q_list[0].soft_bits.item()
                    break
            
            # Calculate BitOps (bits * number of operations)
            num_elements = param.numel()
            total_bitops += param_bits * num_elements
            total_bits += param_bits * num_elements
        
        # Convert to MB for model size
        model_size_mb = total_bits / (8 * 1024 * 1024)
        
        self.quant_metrics['bit_operations'].append(total_bitops)
        self.quant_metrics['model_size'].append(model_size_mb)
        
        return total_bitops, model_size_mb
    
    def save_checkpoint(self, iteration):
        """Save current metrics to disk"""
        # Save raw data
        metrics_path = os.path.join(self.metrics_dir, f'metrics_iter_{iteration}.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump({
                'metrics': self.metrics,
                'quant_metrics': self.quant_metrics,
                'acaq_metrics': self.acaq_metrics,
            }, f)
        
        # Save as CSV for easy access
        self.export_to_csv(iteration)
    
    def export_to_csv(self, iteration):
        """Export metrics to CSV format"""
        # Main metrics
        df_main = pd.DataFrame({
            'iteration': self.metrics['iteration'],
            'time': self.metrics['time'],
            'loss': self.metrics['loss'],
            'psnr': self.metrics['psnr'],
            'avg_bitwidth': self.metrics['avg_bitwidth'] if self.metrics['avg_bitwidth'] else [None] * len(self.metrics['iteration']),
        })
        df_main.to_csv(os.path.join(self.metrics_dir, f'main_metrics_{iteration}.csv'), index=False)
        
        # Quantization metrics
        if self.quant_metrics['embed_bits']:
            df_quant = pd.DataFrame(self.quant_metrics)
            df_quant.to_csv(os.path.join(self.metrics_dir, f'quant_metrics_{iteration}.csv'), index=False)
    
    def plot_training_curves(self, save_path=None):
        """Create publication-quality training curves"""
        if save_path is None:
            save_path = os.path.join(self.metrics_dir, 'training_curves.png')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # PSNR over time
        ax = axes[0, 0]
        ax.plot(self.metrics['time'], self.metrics['psnr'], 'b-', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('PSNR vs Training Time')
        ax.grid(True, alpha=0.3)
        
        # Loss over iterations
        ax = axes[0, 1]
        ax.semilogy(self.metrics['iteration'], self.metrics['loss'], 'r-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)
        
        # Bitwidth evolution
        if self.metrics['avg_bitwidth']:
            ax = axes[1, 0]
            ax.plot(self.metrics['iteration'][:len(self.metrics['avg_bitwidth'])], 
                   self.metrics['avg_bitwidth'], 'g-', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Average Bitwidth')
            ax.set_title('Bitwidth Evolution')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(self.metrics['avg_bitwidth']) + 1)
        
        # Component-wise bitwidths
        if self.metrics['component_bitwidths']:
            ax = axes[1, 1]
            for comp_name, bits_history in self.metrics['component_bitwidths'].items():
                if len(bits_history) > 0:
                    label = comp_name.replace('_', ' ').title()
                    if 'embed' in comp_name.lower():
                        ax.plot(range(len(bits_history)), bits_history, '--', alpha=0.7, label=label)
                    else:
                        ax.plot(range(len(bits_history)), bits_history, '-', alpha=0.7, label=label)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Bitwidth')
            ax.set_title('Component-wise Bitwidth Evolution')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_quantization_analysis(self, save_path=None):
        """Create detailed quantization analysis plots"""
        if save_path is None:
            save_path = os.path.join(self.metrics_dir, 'quantization_analysis.png')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Bitwidth distribution histogram
        if self.metrics['bitwidth_distribution']:
            ax = axes[0, 0]
            latest_dist = self.metrics['bitwidth_distribution'][-1]
            ax.hist(latest_dist, bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Bitwidth')
            ax.set_ylabel('Count')
            ax.set_title('Final Bitwidth Distribution')
            ax.grid(True, alpha=0.3)
        
        # PSNR vs Bitwidth scatter
        if self.metrics['avg_bitwidth'] and len(self.metrics['psnr']) >= len(self.metrics['avg_bitwidth']):
            ax = axes[0, 1]
            psnr_subset = self.metrics['psnr'][:len(self.metrics['avg_bitwidth'])]
            ax.scatter(self.metrics['avg_bitwidth'], psnr_subset, alpha=0.6)
            ax.set_xlabel('Average Bitwidth')
            ax.set_ylabel('PSNR (dB)')
            ax.set_title('PSNR vs Bitwidth Trade-off')
            ax.grid(True, alpha=0.3)
        
        # Model size over time
        if self.quant_metrics['model_size']:
            ax = axes[1, 0]
            iterations = range(len(self.quant_metrics['model_size']))
            ax.plot(iterations, self.quant_metrics['model_size'], 'purple', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Model Size (MB)')
            ax.set_title('Model Compression Over Time')
            ax.grid(True, alpha=0.3)
        
        # Embed vs MLP bits
        if self.quant_metrics['embed_bits'] and self.quant_metrics['mlp_bits']:
            ax = axes[1, 1]
            iterations = range(len(self.quant_metrics['embed_bits']))
            ax.plot(iterations, self.quant_metrics['embed_bits'], 'b-', label='Embeddings', linewidth=2)
            ax.plot(iterations, self.quant_metrics['mlp_bits'], 'r-', label='MLP', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Average Bitwidth')
            ax.set_title('Component-wise Compression')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_table(self):
        """Generate LaTeX table for paper"""
        summary = {
            'Metric': [],
            'Baseline': [],
            'Quantized (8-bit)': [],
            'A-CAQ': []
        }
        
        # Calculate summary statistics
        if self.metrics['psnr']:
            summary['Metric'].append('Final PSNR (dB)')
            summary['Baseline'].append('N/A')
            summary['Quantized (8-bit)'].append(f"{self.metrics['psnr'][1000]:.2f}" if len(self.metrics['psnr']) > 1000 else 'N/A')
            summary['A-CAQ'].append(f"{self.metrics['psnr'][-1]:.2f}")
        
        if self.metrics['avg_bitwidth']:
            summary['Metric'].append('Average Bitwidth')
            summary['Baseline'].append('32.0')
            summary['Quantized (8-bit)'].append('8.0')
            summary['A-CAQ'].append(f"{self.metrics['avg_bitwidth'][-1]:.2f}")
        
        if self.quant_metrics['model_size']:
            summary['Metric'].append('Model Size (MB)')
            summary['Baseline'].append('N/A')
            summary['Quantized (8-bit)'].append('N/A')
            summary['A-CAQ'].append(f"{self.quant_metrics['model_size'][-1]:.2f}")
        
        # Save as CSV
        df = pd.DataFrame(summary)
        df.to_csv(os.path.join(self.metrics_dir, 'summary_table.csv'), index=False)
        
        # Generate LaTeX
        latex_table = df.to_latex(index=False, float_format="%.2f")
        with open(os.path.join(self.metrics_dir, 'summary_table.tex'), 'w') as f:
            f.write(latex_table)
        
        return df