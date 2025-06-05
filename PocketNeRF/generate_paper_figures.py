import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_metrics_history(experiment_path):
    """Load metrics history from experiment"""
    with open(experiment_path / 'all_metrics_history.pkl', 'rb') as f:
        return pickle.load(f)

def plot_psnr_curves(baseline_metrics, method_metrics, save_path):
    """Plot train and test PSNR curves for both methods"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Test PSNR vs Time
    ax1.plot(baseline_metrics['test']['time'], baseline_metrics['test']['psnr'], 
             'b-', label='Baseline', linewidth=2)
    ax1.plot(method_metrics['test']['time'], method_metrics['test']['psnr'], 
             'r-', label='Our Method', linewidth=2)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Test PSNR (dB)')
    ax1.set_title('Test PSNR vs Training Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Test PSNR vs Iteration
    ax2.plot(baseline_metrics['test']['iter'], baseline_metrics['test']['psnr'], 
             'b-', label='Baseline', linewidth=2)
    ax2.plot(method_metrics['test']['iter'], method_metrics['test']['psnr'], 
             'r-', label='Our Method', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Test PSNR (dB)')
    ax2.set_title('Test PSNR vs Iteration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'psnr_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_metrics(baseline_metrics, method_metrics, save_path):
    """Plot PSNR, SSIM, and LPIPS curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics_info = [
        ('psnr', 'PSNR (dB)', True),  # higher is better
        ('ssim', 'SSIM', True),
        ('lpips', 'LPIPS', False)  # lower is better
    ]
    
    for ax, (metric, ylabel, higher_better) in zip(axes, metrics_info):
        baseline_vals = baseline_metrics['test'][metric]
        method_vals = method_metrics['test'][metric]
        iters = baseline_metrics['test']['iter']
        
        ax.plot(iters, baseline_vals, 'b-', label='Baseline', linewidth=2)
        ax.plot(iters, method_vals, 'r-', label='Our Method', linewidth=2)
        
        # Highlight improvement region
        if higher_better:
            improvement = np.array(method_vals) > np.array(baseline_vals)
        else:
            improvement = np.array(method_vals) < np.array(baseline_vals)
        
        if np.any(improvement):
            ax.fill_between(iters, baseline_vals, method_vals, 
                           where=improvement, alpha=0.2, color='green')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs Iteration')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'all_metrics_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_vs_test_psnr(baseline_metrics, method_metrics, save_path):
    """Plot both training and test PSNR to show generalization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Baseline - only plot if we have training data
    if 'train' in baseline_metrics and baseline_metrics['train']['iter']:
        ax1.plot(baseline_metrics['train']['iter'], baseline_metrics['train']['psnr'], 
                 'b--', label='Train', linewidth=2, alpha=0.7)
    ax1.plot(baseline_metrics['test']['iter'], baseline_metrics['test']['psnr'], 
             'b-', label='Test', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Baseline: Train vs Test PSNR')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Your method - only plot if we have training data
    if 'train' in method_metrics and method_metrics['train']['iter']:
        ax2.plot(method_metrics['train']['iter'], method_metrics['train']['psnr'], 
                 'r--', label='Train', linewidth=2, alpha=0.7)
    ax2.plot(method_metrics['test']['iter'], method_metrics['test']['psnr'], 
             'r-', label='Test', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Our Method: Train vs Test PSNR')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'train_vs_test_psnr.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_memory_usage(baseline_metrics, method_metrics, save_path):
    """Plot memory usage over time"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    has_data = False
    
    if 'memory' in baseline_metrics and baseline_metrics['memory']['iter']:
        ax.plot(baseline_metrics['memory']['iter'], 
                baseline_metrics['memory']['allocated_gb'], 
                'b-', label='Baseline', linewidth=2)
        has_data = True
    
    if 'memory' in method_metrics and method_metrics['memory']['iter']:
        ax.plot(method_metrics['memory']['iter'], 
                method_metrics['memory']['allocated_gb'], 
                'r-', label='Our Method', linewidth=2)
        has_data = True
    
    if has_data:
        ax.set_xlabel('Iteration')
        ax.set_ylabel('GPU Memory (GB)')
        ax.set_title('GPU Memory Usage During Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'memory_usage.pdf', dpi=300, bbox_inches='tight')
    
    plt.close()

def generate_summary_stats(baseline_report, method_report, save_path):
    """Generate a text file with key statistics for the paper"""
    baseline_final = baseline_report['final_test_metrics']
    method_final = method_report['final_test_metrics']
    
    psnr_improvement = method_final['psnr'] - baseline_final['psnr']
    ssim_improvement = method_final['ssim'] - baseline_final['ssim']
    lpips_improvement = baseline_final['lpips'] - method_final['lpips']  # Lower is better
    
    baseline_hours = baseline_report['total_time_seconds'] / 3600
    method_hours = method_report['total_time_seconds'] / 3600
    speedup = baseline_report['total_time_seconds'] / method_report['total_time_seconds']
    
    stats = f"""Experiment Summary
==================

Training Details:
- Total iterations: {baseline_report['total_iterations']}
- Baseline training time: {baseline_hours:.2f} hours ({baseline_report['total_time_seconds']:.0f} seconds)
- Our method training time: {method_hours:.2f} hours ({method_report['total_time_seconds']:.0f} seconds)
- Speedup: {speedup:.2f}x

Final Test Metrics:
- Baseline PSNR: {baseline_final['psnr']:.2f} ± {baseline_final['std_psnr']:.2f} dB
- Our PSNR: {method_final['psnr']:.2f} ± {method_final['std_psnr']:.2f} dB
- PSNR improvement: {psnr_improvement:+.2f} dB

- Baseline SSIM: {baseline_final['ssim']:.3f} ± {baseline_final['std_ssim']:.3f}
- Our SSIM: {method_final['ssim']:.3f} ± {method_final['std_ssim']:.3f}
- SSIM improvement: {ssim_improvement:+.3f}

- Baseline LPIPS: {baseline_final['lpips']:.3f} ± {baseline_final['std_lpips']:.3f}
- Our LPIPS: {method_final['lpips']:.3f} ± {method_final['std_lpips']:.3f}
- LPIPS improvement: {lpips_improvement:+.3f} (positive = better)

Memory Usage:
- Baseline peak memory: {baseline_report.get('peak_memory_gb', 0):.2f} GB
- Our peak memory: {method_report.get('peak_memory_gb', 0):.2f} GB

Key Results for Paper:
- {'BETTER' if psnr_improvement > 0 else 'WORSE'} by {abs(psnr_improvement):.2f} dB PSNR
- {'BETTER' if ssim_improvement > 0 else 'WORSE'} by {abs(ssim_improvement):.3f} SSIM
- {'BETTER' if lpips_improvement > 0 else 'WORSE'} by {abs(lpips_improvement):.3f} LPIPS
- {'FASTER' if speedup > 1 else 'SLOWER'} by {speedup:.2f}x
"""
    
    with open(save_path / 'summary_stats.txt', 'w') as f:
        f.write(stats)
    
    print(stats)

def generate_latex_table(baseline_report, method_report, save_path):
    """Generate LaTeX table with final metrics"""
    baseline_metrics = baseline_report['final_test_metrics']
    method_metrics = method_report['final_test_metrics']
    
    # Determine which metrics are better
    psnr_better = method_metrics['psnr'] > baseline_metrics['psnr']
    ssim_better = method_metrics['ssim'] > baseline_metrics['ssim']
    lpips_better = method_metrics['lpips'] < baseline_metrics['lpips']
    
    # Format with bold for better metrics
    def format_metric(value, std, is_better, decimals=2):
        if is_better:
            return f"\\mathbf{{{value:.{decimals}f}}} \\pm {std:.{decimals}f}"
        else:
            return f"{value:.{decimals}f} \\pm {std:.{decimals}f}"
    
    latex = f"""
\\begin{{table}}[h]
\\centering
\\begin{{tabular}}{{lccc}}
\\toprule
Method & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ \\\\
\\midrule
Baseline & ${baseline_metrics['psnr']:.2f} \\pm {baseline_metrics['std_psnr']:.2f}$ & ${baseline_metrics['ssim']:.3f} \\pm {baseline_metrics['std_ssim']:.3f}$ & ${baseline_metrics['lpips']:.3f} \\pm {baseline_metrics['std_lpips']:.3f}$ \\\\
Ours & ${format_metric(method_metrics['psnr'], method_metrics['std_psnr'], psnr_better, 2)}$ & ${format_metric(method_metrics['ssim'], method_metrics['std_ssim'], ssim_better, 3)}$ & ${format_metric(method_metrics['lpips'], method_metrics['std_lpips'], lpips_better, 3)}$ \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Quantitative comparison on test views. Best results in \\textbf{{bold}}.}}
\\label{{tab:quantitative}}
\\end{{table}}
"""
    
    with open(save_path / 'results_table.tex', 'w') as f:
        f.write(latex)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, required=True, 
                       help='Path to baseline experiment')
    parser.add_argument('--method', type=str, required=True,
                       help='Path to your method experiment')
    parser.add_argument('--output', type=str, default='./paper_figures',
                       help='Output directory for figures')
    args = parser.parse_args()
    
    # Load data
    baseline_path = Path(args.baseline)
    method_path = Path(args.method)
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Check if required files exist
    if not (baseline_path / 'all_metrics_history.pkl').exists():
        print(f"Error: {baseline_path / 'all_metrics_history.pkl'} not found!")
        print("Make sure the experiment was run with comprehensive evaluation enabled.")
        exit(1)
    
    if not (method_path / 'all_metrics_history.pkl').exists():
        print(f"Error: {method_path / 'all_metrics_history.pkl'} not found!")
        print("Make sure the experiment was run with comprehensive evaluation enabled.")
        exit(1)
    
    # Load metrics
    baseline_metrics = load_metrics_history(baseline_path)
    method_metrics = load_metrics_history(method_path)
    
    with open(baseline_path / 'final_report.json', 'r') as f:
        baseline_report = json.load(f)
    with open(method_path / 'final_report.json', 'r') as f:
        method_report = json.load(f)
    
    # Generate all figures and analyses
    print("Generating PSNR curves...")
    plot_psnr_curves(baseline_metrics, method_metrics, output_path)
    
    print("Generating all metrics curves...")
    plot_all_metrics(baseline_metrics, method_metrics, output_path)
    
    print("Generating train vs test comparison...")
    plot_training_vs_test_psnr(baseline_metrics, method_metrics, output_path)
    
    print("Generating memory usage plot...")
    plot_memory_usage(baseline_metrics, method_metrics, output_path)
    
    print("Generating LaTeX table...")
    generate_latex_table(baseline_report, method_report, output_path)
    
    print("\nGenerating summary statistics...")
    generate_summary_stats(baseline_report, method_report, output_path)
    
    print(f"\nAll figures and analyses saved to {output_path}/")
    print("\nGenerated files:")
    print("- psnr_curves.pdf: Test PSNR vs time and iteration")
    print("- all_metrics_curves.pdf: PSNR, SSIM, and LPIPS comparison")
    print("- train_vs_test_psnr.pdf: Training vs test PSNR (generalization)")
    print("- memory_usage.pdf: GPU memory consumption")
    print("- results_table.tex: LaTeX table for paper")
    print("- summary_stats.txt: Key statistics and improvements")