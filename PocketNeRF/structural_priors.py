"""
Structural Priors for PocketNeRF
Implementation of planarity constraints and Manhattan-world assumptions for indoor scene reconstruction.
Based on the research described in the PocketNeRF milestone document.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


def depth_prior_loss(depth_pred: torch.Tensor, depth_prior: torch.Tensor, 
                    weight: float = 1.0, use_ranking: bool = True) -> torch.Tensor:
    """
    Depth prior loss as described in the research preview.
    
    Args:
        depth_pred: Predicted depth values [N_rays]
        depth_prior: Prior depth estimates [N_rays] 
        weight: Loss weight
        use_ranking: Whether to use ranking loss instead of MSE
    
    Returns:
        Depth prior loss
    """
    if use_ranking:
        # Use ranking loss to enforce relative depth ordering
        # Sample pairs of pixels and enforce depth ordering
        n_pairs = min(1000, depth_pred.shape[0] // 2)
        if n_pairs > 0:
            idx1 = torch.randint(0, depth_pred.shape[0], (n_pairs,))
            idx2 = torch.randint(0, depth_pred.shape[0], (n_pairs,))
            
            # Get depth differences
            depth_diff_pred = depth_pred[idx1] - depth_pred[idx2]
            depth_diff_prior = depth_prior[idx1] - depth_prior[idx2]
            
            # Hinge loss to enforce same ordering
            loss = F.relu(-depth_diff_pred * depth_diff_prior.sign()).mean()
            return weight * loss
    else:
        # Simple MSE loss (normalize depths first)
        depth_pred_norm = (depth_pred - depth_pred.mean()) / (depth_pred.std() + 1e-8)
        depth_prior_norm = (depth_prior - depth_prior.mean()) / (depth_prior.std() + 1e-8)
        return weight * F.mse_loss(depth_pred_norm, depth_prior_norm)
    
    return torch.tensor(0.0, device=depth_pred.device)


def planarity_loss_improved(depth_map: torch.Tensor, rays_d: torch.Tensor,
                          weight: float = 1.0, patch_size: int = 8, 
                          depth_threshold: float = 0.05, min_patch_size: int = 4) -> torch.Tensor:
    """
    Improved planarity constraints using local depth consistency instead of random sampling.
    This reduces overfitting by providing more stable, spatially-coherent constraints.
    
    Args:
        depth_map: Rendered depth map [N_rays]
        rays_d: Ray directions [N_rays, 3]
        weight: Loss weight
        patch_size: Size of local patches to analyze
        depth_threshold: Threshold for depth variation within patches
        min_patch_size: Minimum number of rays needed for patch analysis
        
    Returns:
        Planarity loss
    """
    device = depth_map.device
    N_rays = depth_map.shape[0]
    
    if N_rays < min_patch_size:
        return torch.tensor(0.0, device=device)
    
    # For few-shot training, use simpler local smoothness instead of complex plane fitting
    # This is more stable and generalizes better
    
    # Method 1: Local depth smoothness with edge preservation
    n_pairs = min(500, N_rays // 4)  # Reduced from previous implementation
    if n_pairs > 0:
        # Sample neighboring rays (more likely to be spatially related)
        idx1 = torch.randint(0, N_rays-1, (n_pairs,), device=device)
        idx2 = idx1 + 1  # Adjacent rays instead of completely random
        
        depth1 = depth_map[idx1]
        depth2 = depth_map[idx2]
        
        # Edge-preserving smoothness: smaller penalty for large depth differences
        # This allows for depth discontinuities while encouraging smoothness
        depth_diff = torch.abs(depth1 - depth2)
        smoothness_loss = torch.mean(torch.exp(-depth_diff / depth_threshold) * depth_diff)
        
        return weight * smoothness_loss
    
    return torch.tensor(0.0, device=device)


def planarity_loss(points: torch.Tensor, normals: torch.Tensor, 
                  depth_map: torch.Tensor, rays_d: torch.Tensor,
                  weight: float = 1.0, plane_threshold: float = 0.1,
                  smoothness_weight: float = 0.1) -> torch.Tensor:
    """
    DEPRECATED: Using improved planarity loss instead.
    Kept for compatibility but redirects to improved version.
    """
    return planarity_loss_improved(depth_map, rays_d, weight, 
                                 depth_threshold=plane_threshold)


def manhattan_world_loss_adaptive(normals: torch.Tensor, depth_map: torch.Tensor,
                                weight: float = 1.0, confidence_threshold: float = 0.7,
                                adaptive_threshold: float = 0.8) -> torch.Tensor:
    """
    Adaptive Manhattan-world assumption that only applies to surfaces that actually appear axis-aligned.
    This reduces overfitting by not forcing non-Manhattan surfaces to be axis-aligned.
    
    Args:
        normals: Surface normals [N_rays, 3]
        depth_map: Depth values for confidence weighting [N_rays]
        weight: Loss weight
        confidence_threshold: Minimum alignment confidence to apply constraint
        adaptive_threshold: Threshold for considering a normal axis-aligned
        
    Returns:
        Adaptive Manhattan world loss
    """
    if normals.dim() == 3:
        normals = normals[:, -1, :]  # [N_rays, 3]
    
    device = normals.device
    
    # Define canonical Manhattan directions
    manhattan_dirs = torch.tensor([
        [1.0, 0.0, 0.0],  # X-axis
        [0.0, 1.0, 0.0],  # Y-axis  
        [0.0, 0.0, 1.0],  # Z-axis
    ], device=device, dtype=normals.dtype)
    
    # Normalize normals
    normals_norm = F.normalize(normals, dim=-1)
    
    # Compute alignment with each Manhattan direction
    alignments = torch.abs(torch.matmul(normals_norm, manhattan_dirs.T))  # [N_rays, 3]
    best_alignment, best_idx = torch.max(alignments, dim=-1)  # [N_rays]
    
    # Only apply Manhattan constraint to normals that are already somewhat aligned
    # This prevents forcing curved or diagonal surfaces to be axis-aligned
    confident_mask = best_alignment > adaptive_threshold
    
    if torch.sum(confident_mask) == 0:
        return torch.tensor(0.0, device=device)
    
    # Apply loss only to confident predictions
    confident_alignments = best_alignment[confident_mask]
    manhattan_loss = (1.0 - confident_alignments).mean()
    
    # Weight by number of confident predictions to avoid over-penalizing
    confidence_ratio = torch.sum(confident_mask).float() / normals.shape[0]
    
    return weight * manhattan_loss * confidence_ratio


def manhattan_world_loss(normals: torch.Tensor, weight: float = 1.0,
                        confidence_threshold: float = 0.5) -> torch.Tensor:
    """
    DEPRECATED: Using adaptive Manhattan world loss instead.
    Kept for compatibility but redirects to adaptive version.
    """
    if normals.dim() == 3:
        normals = normals[:, -1, :]
    
    # Use adaptive version with depth map as None (will use equal weighting)
    depth_dummy = torch.ones(normals.shape[0], device=normals.device)
    return manhattan_world_loss_adaptive(normals, depth_dummy, weight)


def normal_consistency_loss_improved(normals_pred: torch.Tensor, 
                                   depth_map: torch.Tensor,
                                   weight: float = 1.0,
                                   spatial_weight: float = 0.5) -> torch.Tensor:
    """
    Improved normal consistency that uses depth-based confidence weighting.
    
    Args:
        normals_pred: Predicted normals [N_rays, 3]
        depth_map: Depth values for confidence weighting [N_rays]
        weight: Loss weight
        spatial_weight: Weight for spatial consistency vs random consistency
        
    Returns:
        Improved normal consistency loss
    """
    device = normals_pred.device
    N_rays = normals_pred.shape[0]
    
    if N_rays <= 1:
        return torch.tensor(0.0, device=device)
    
    # Normalize normals
    normals_norm = F.normalize(normals_pred, dim=-1)
    
    # Method 1: Spatially local consistency (adjacent rays)
    spatial_loss = torch.tensor(0.0, device=device)
    if N_rays > 1:
        n_spatial = min(100, N_rays - 1)
        idx1 = torch.randint(0, N_rays-1, (n_spatial,), device=device)
        idx2 = idx1 + 1  # Adjacent rays
        
        normal1 = normals_norm[idx1]
        normal2 = normals_norm[idx2]
        
        # Weight by depth similarity (nearby depths should have similar normals)
        depth1 = depth_map[idx1]
        depth2 = depth_map[idx2]
        depth_similarity = torch.exp(-torch.abs(depth1 - depth2))
        
        cosine_sim = torch.sum(normal1 * normal2, dim=-1)
        spatial_loss = torch.mean(depth_similarity * (1.0 - cosine_sim))
    
    # Method 2: Random consistency (reduced weight)
    random_loss = torch.tensor(0.0, device=device)
    if N_rays > 1:
        n_random = min(50, N_rays // 2)  # Reduced from original
        idx1 = torch.randint(0, N_rays, (n_random,), device=device)
        idx2 = torch.randint(0, N_rays, (n_random,), device=device)
        
        normal1 = normals_norm[idx1]
        normal2 = normals_norm[idx2]
        
        cosine_sim = torch.sum(normal1 * normal2, dim=-1)
        random_loss = (1.0 - cosine_sim).mean()
    
    total_loss = spatial_weight * spatial_loss + (1.0 - spatial_weight) * random_loss
    return weight * total_loss


def normal_consistency_loss(normals_pred: torch.Tensor, normals_prior: Optional[torch.Tensor] = None,
                          weight: float = 1.0) -> torch.Tensor:
    """
    DEPRECATED: Using improved normal consistency loss instead.
    Kept for compatibility but redirects to improved version.
    """
    if normals_prior is not None:
        # Use original implementation for prior-based consistency
        device = normals_pred.device
        normals_pred_norm = F.normalize(normals_pred, dim=-1)
        normals_prior_norm = F.normalize(normals_prior, dim=-1)
        
        cosine_sim = torch.sum(normals_pred_norm * normals_prior_norm, dim=-1)
        consistency_loss = (1.0 - cosine_sim).mean()
        
        return weight * consistency_loss
    else:
        # Use improved version with depth-based weighting (dummy depth)
        depth_dummy = torch.ones(normals_pred.shape[0], device=normals_pred.device)
        return normal_consistency_loss_improved(normals_pred, depth_dummy, weight)


def detect_planar_regions(depth_map: torch.Tensor, height: int, width: int,
                         patch_size: int = 8, plane_threshold: float = 0.05) -> torch.Tensor:
    """
    Simple plane detection using depth map patches.
    
    Args:
        depth_map: Rendered depth map [N_rays]
        height: Image height
        width: Image width  
        patch_size: Size of patches for plane detection
        plane_threshold: Threshold for planarity
        
    Returns:
        Binary mask indicating planar regions [N_rays]
    """
    device = depth_map.device
    
    # Reshape depth map to image
    if depth_map.shape[0] == height * width:
        depth_img = depth_map.view(height, width)
    else:
        # Handle random ray sampling - return all zeros
        return torch.zeros_like(depth_map, dtype=torch.bool, device=device)
    
    planar_mask = torch.zeros_like(depth_img, dtype=torch.bool, device=device)
    
    # Check each patch
    for i in range(0, height - patch_size + 1, patch_size // 2):
        for j in range(0, width - patch_size + 1, patch_size // 2):
            patch = depth_img[i:i+patch_size, j:j+patch_size]
            
            # Check if patch is approximately planar
            patch_var = torch.var(patch)
            if patch_var < plane_threshold:
                planar_mask[i:i+patch_size, j:j+patch_size] = True
    
    return planar_mask.view(-1)


def combine_structural_losses(depth_pred: torch.Tensor, 
                            points: torch.Tensor,
                            normals: Optional[torch.Tensor] = None,
                            rays_d: torch.Tensor = None,
                            depth_prior: Optional[torch.Tensor] = None,
                            height: int = None, width: int = None,
                            weights: dict = None) -> Tuple[torch.Tensor, dict]:
    """
    Combine all structural losses for PocketNeRF with improved overfitting resistance.
    
    Args:
        depth_pred: Predicted depth [N_rays]
        points: 3D points [N_rays, N_samples, 3]
        normals: Surface normals [N_rays, 3] (optional)
        rays_d: Ray directions [N_rays, 3]  
        depth_prior: Prior depth estimates [N_rays] (optional)
        height: Image height (for plane detection)
        width: Image width (for plane detection)
        weights: Dictionary of loss weights
        
    Returns:
        total_loss: Combined structural loss
        loss_dict: Dictionary of individual losses for logging
    """
    if weights is None:
        weights = {
            'depth_prior': 1.0,
            'planarity': 1.0, 
            'manhattan': 1.0,
            'normal_consistency': 0.5
        }
    
    device = depth_pred.device
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device)
    
    # Depth prior loss
    if depth_prior is not None and 'depth_prior' in weights:
        depth_loss = depth_prior_loss(depth_pred, depth_prior, weights['depth_prior'])
        loss_dict['depth_prior'] = depth_loss
        total_loss += depth_loss
    
    # Improved planarity loss  
    if 'planarity' in weights and rays_d is not None:
        planar_loss = planarity_loss_improved(depth_pred, rays_d, weights['planarity'])
        loss_dict['planarity'] = planar_loss
        total_loss += planar_loss
    
    # Adaptive Manhattan world loss
    if normals is not None and 'manhattan' in weights:
        manhattan_loss = manhattan_world_loss_adaptive(normals, depth_pred, weights['manhattan'])
        loss_dict['manhattan'] = manhattan_loss
        total_loss += manhattan_loss
    
    # Improved normal consistency loss
    if normals is not None and 'normal_consistency' in weights:
        normal_loss = normal_consistency_loss_improved(normals, depth_pred, weights['normal_consistency'])
        loss_dict['normal_consistency'] = normal_loss  
        total_loss += normal_loss
    
    return total_loss, loss_dict 