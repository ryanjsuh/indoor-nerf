import torch
import numpy as np
import lpips
from skimage.metrics import structural_similarity as ssim
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import time

class ComprehensiveEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.metrics_history = {
            'train': {'iter': [], 'psnr': [], 'time': []},
            'test': {'iter': [], 'psnr': [], 'ssim': [], 'lpips': [], 'time': []},
            'memory': {'iter': [], 'allocated_gb': [], 'reserved_gb': []}
        }
        self.start_time = time.time()
    
    def compute_metrics(self, pred, target):
        # PSNR
        mse = torch.mean((pred - target) ** 2)
        psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]))
        
        # Change to numpy
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        # SSIM
        ssim_val = ssim(target_np, pred_np, channel_axis=2, data_range=1.0)
        
        # LPIPS
        pred_lpips = pred.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        target_lpips = target.permute(2, 0, 1).unsqueeze(0) * 2 - 1
        lpips_val = self.lpips_fn(pred_lpips, target_lpips).item()
        
        return {
            'psnr': psnr.item(),
            'ssim': ssim_val,
            'lpips': lpips_val
        }
    
    def evaluate_test_set(self, model_fn, poses, hwf, K, chunk, render_kwargs, images_gt):
        """Evaluate model on entire test set"""
        H, W, focal = hwf
        all_metrics = []
        all_preds = []
        
        for i, (pose, gt_img) in enumerate(zip(poses, images_gt)):
            with torch.no_grad():
                rgb, depth, acc, _ = model_fn(H, W, K, chunk=chunk, 
                                              c2w=pose[:3,:4], **render_kwargs)
                
                
                if not isinstance(gt_img, torch.Tensor):
                    gt_img = torch.Tensor(gt_img).to(rgb.device)
                
                metrics = self.compute_metrics(rgb, gt_img)
                all_metrics.append(metrics)
                all_preds.append(rgb.cpu().numpy())
        
        
        avg_metrics = {
            'psnr': np.mean([m['psnr'] for m in all_metrics]),
            'ssim': np.mean([m['ssim'] for m in all_metrics]),
            'lpips': np.mean([m['lpips'] for m in all_metrics]),
            'std_psnr': np.std([m['psnr'] for m in all_metrics]),
            'std_ssim': np.std([m['ssim'] for m in all_metrics]),
            'std_lpips': np.std([m['lpips'] for m in all_metrics])
        }
        
        return avg_metrics, all_preds, all_metrics
    
    def record_test_metrics(self, iteration, test_metrics):
        """Record test metrics for plotting"""
        elapsed_time = time.time() - self.start_time
        self.metrics_history['test']['iter'].append(iteration)
        self.metrics_history['test']['psnr'].append(test_metrics['psnr'])
        self.metrics_history['test']['ssim'].append(test_metrics['ssim'])
        self.metrics_history['test']['lpips'].append(test_metrics['lpips'])
        self.metrics_history['test']['time'].append(elapsed_time)
    
    def record_memory_usage(self, iteration):
        """Record GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9  
            reserved = torch.cuda.memory_reserved() / 1e9   
            self.metrics_history['memory']['iter'].append(iteration)
            self.metrics_history['memory']['allocated_gb'].append(allocated)
            self.metrics_history['memory']['reserved_gb'].append(reserved)
    
    def save_metrics(self, save_path):
        """Save all metrics to file"""
        with open(save_path, 'wb') as f:
            pickle.dump(self.metrics_history, f)
    
    def generate_comparison_figures(self, baseline_preds, method_preds, gt_images, save_dir):
        """Generate visual comparison figures"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        n_views = len(gt_images)
        selected_indices = [0, n_views//2, n_views-1] + \
                          list(np.random.choice(range(1, n_views-1), min(2, n_views-3), replace=False))
        
        for idx in selected_indices:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            axes[0, 0].imshow(baseline_preds[idx])
            axes[0, 0].set_title('Baseline')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(method_preds[idx])
            axes[0, 1].set_title('Our Method')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(gt_images[idx])
            axes[0, 2].set_title('Ground Truth')
            axes[0, 2].axis('off')
            
            baseline_error = np.abs(baseline_preds[idx] - gt_images[idx])
            method_error = np.abs(method_preds[idx] - gt_images[idx])
            
            im1 = axes[1, 0].imshow(baseline_error.mean(axis=2), cmap='hot', vmin=0, vmax=0.2)
            axes[1, 0].set_title('Baseline Error')
            axes[1, 0].axis('off')
            
            im2 = axes[1, 1].imshow(method_error.mean(axis=2), cmap='hot', vmin=0, vmax=0.2)
            axes[1, 1].set_title('Our Method Error')
            axes[1, 1].axis('off')
            
            axes[1, 2].imshow(baseline_error.mean(axis=2) - method_error.mean(axis=2), 
                             cmap='RdBu', vmin=-0.1, vmax=0.1)
            axes[1, 2].set_title('Error Difference (Blue=Ours Better)')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_dir / f'comparison_view_{idx:03d}.png', dpi=150, bbox_inches='tight')
            plt.close()