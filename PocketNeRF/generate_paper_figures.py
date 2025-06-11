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

def plot_metrics_over_time(metrics, save_path):
    """Plot all metrics over time for a single experiment"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Test PSNR vs Iteration (keep as line plot)
    ax = axes[0]
    if 'train' in metrics and metrics['train']['iter']:
        ax.plot(metrics['train']['iter'], metrics['train']['psnr'], 
                'g--', label='Train', linewidth=2, alpha=0.7)
    ax.plot(metrics['test']['iter'], metrics['test']['psnr'], 
            'b-', label='Test', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Train vs Test PSNR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Test SSIM vs Iteration (scatter plot)
    ax = axes[1]
    ax.scatter(metrics['test']['iter'], metrics['test']['ssim'], 
               color='r', s=50, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('SSIM')
    ax.set_title('Test SSIM over Training')
    ax.grid(True, alpha=0.3)
    
    # Test LPIPS vs Iteration (scatter plot)
    ax = axes[2]
    ax.scatter(metrics['test']['iter'], metrics['test']['lpips'], 
               color='m', s=50, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('LPIPS')
    ax.set_title('Test LPIPS over Training')
    ax.grid(True, alpha=0.3)
    
    # Memory Usage (scatter plot)
    ax = axes[3]
    if 'memory' in metrics and metrics['memory']['iter']:
        ax.scatter(metrics['memory']['iter'], 
                   metrics['memory']['allocated_gb'], 
                   color='c', s=50, alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('GPU Memory (GB)')
        ax.set_title('GPU Memory Usage')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No memory data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('GPU Memory Usage')
    
    plt.tight_layout()
    plt.savefig(save_path / 'metrics_overview.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def generate_single_experiment_summary(report, metrics, save_path):
    """Generate summary for a single experiment"""
    if report:
        final_metrics = report['final_test_metrics']
        total_hours = report['total_time_seconds'] / 3600
        iterations = report.get('total_iterations', 8000)
        
        summary = f"""Single Experiment Summary
=========================
Experiment: {report['experiment']}

Training Details:
- Total iterations: {iterations}
- Training time: {total_hours:.2f} hours ({report['total_time_seconds']:.0f} seconds)
- Peak GPU memory: {report.get('peak_memory_gb', 0):.2f} GB

Final Test Metrics (at iteration {iterations}):
- PSNR: {final_metrics['psnr']:.2f} ± {final_metrics['std_psnr']:.2f} dB
- SSIM: {final_metrics['ssim']:.3f} ± {final_metrics['std_ssim']:.3f}
- LPIPS: {final_metrics['lpips']:.3f} ± {final_metrics['std_lpips']:.3f}

Key Configuration:
- Learning rate: {report['config'].get('lrate', 'N/A')}
- Batch size: {report['config'].get('N_rand', 'N/A')}
- Network depth: {report['config'].get('netdepth', 'N/A')}
- Network width: {report['config'].get('netwidth', 'N/A')}
"""
    else:
        # Fallback if no final report
        test_psnr_final = metrics['test']['psnr'][-1] if metrics['test']['psnr'] else 0
        test_ssim_final = metrics['test']['ssim'][-1] if metrics['test']['ssim'] else 0
        test_lpips_final = metrics['test']['lpips'][-1] if metrics['test']['lpips'] else 0
        
        summary = f"""Single Experiment Summary
=========================

Final Test Metrics (from metrics history):
- PSNR: {test_psnr_final:.2f} dB
- SSIM: {test_ssim_final:.3f}
- LPIPS: {test_lpips_final:.3f}

Note: final_report.json not available, showing last recorded values.
"""
    
    print(summary)
    with open(save_path / 'experiment_summary.txt', 'w') as f:
        f.write(summary)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, 
                       help='Path to experiment directory')
    parser.add_argument('--output', type=str, default='./experiment_analysis',
                       help='Output directory for analysis')
    args = parser.parse_args()
    
    exp_path = Path(args.experiment)
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Check if required files exist
    if not (exp_path / 'all_metrics_history.pkl').exists():
        print(f"Error: {exp_path / 'all_metrics_history.pkl'} not found!")
        exit(1)
    
    # Load data
    metrics = load_metrics_history(exp_path)
    
    # Try to load final report
    report = None
    if (exp_path / 'final_report.json').exists():
        try:
            with open(exp_path / 'final_report.json', 'r') as f:
                report = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: final_report.json is corrupted: {e}")
            print("Continuing with metrics from all_metrics_history.pkl...")
            report = None
    
    # Generate analysis
    print("Generating metrics plots...")
    plot_metrics_over_time(metrics, output_path)
    
    print("\nGenerating summary...")
    generate_single_experiment_summary(report, metrics, output_path)
    
    print(f"\nAnalysis saved to {output_path}/")
    print("Generated files:")
    print("- metrics_overview.pdf: All metrics over training")
    print("- experiment_summary.txt: Summary statistics")