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


def planarity_loss(points: torch.Tensor, normals: torch.Tensor, 
                  depth_map: torch.Tensor, rays_d: torch.Tensor,
                  weight: float = 1.0, plane_threshold: float = 0.1,
                  smoothness_weight: float = 0.1) -> torch.Tensor:
    """
    Planarity constraints for indoor scenes.
    Encourages planar surfaces in regions identified as walls, floors, ceilings.
    
    Args:
        points: 3D points sampled along rays [N_rays, N_samples, 3]
        normals: Surface normals (if available) [N_rays, N_samples, 3]  
        depth_map: Rendered depth map [N_rays]
        rays_d: Ray directions [N_rays, 3]
        weight: Loss weight
        plane_threshold: Threshold for plane fitting
        smoothness_weight: Weight for depth smoothness regularization
        
    Returns:
        Planarity loss
    """
    device = points.device
    N_rays = points.shape[0]
    
    if N_rays < 3:
        return torch.tensor(0.0, device=device)
    
    # Convert depth map to 3D points on surface
    # Assume we have the final depth per ray
    surface_points = rays_d * depth_map.unsqueeze(-1)  # [N_rays, 3]
    
    total_loss = torch.tensor(0.0, device=device)
    n_planes = 0
    
    # Sample random subsets of points and fit planes
    n_trials = min(50, N_rays // 4)  # Adaptive number of trials
    
    for _ in range(n_trials):
        # Randomly sample 3 points to define a plane
        if N_rays >= 3:
            indices = torch.randperm(N_rays, device=device)[:3]
            plane_points = surface_points[indices]  # [3, 3]
            
            # Fit plane: compute normal vector
            v1 = plane_points[1] - plane_points[0]
            v2 = plane_points[2] - plane_points[0]
            normal = torch.cross(v1, v2)
            normal_norm = torch.norm(normal)
            
            if normal_norm > 1e-6:
                normal = normal / normal_norm
                
                # Find points close to this plane
                plane_origin = plane_points[0]
                distances = torch.abs(torch.sum((surface_points - plane_origin) * normal, dim=-1))
                
                # Points within threshold are considered on the plane
                on_plane_mask = distances < plane_threshold
                n_on_plane = torch.sum(on_plane_mask)
                
                if n_on_plane >= 3:  # Need at least 3 points for a meaningful plane
                    # Penalize deviation from planarity
                    plane_loss = distances[on_plane_mask].mean()
                    total_loss += weight * plane_loss
                    n_planes += 1
    
    # Add depth smoothness regularization
    if N_rays > 1:
        # Sample neighboring rays and penalize large depth discontinuities
        n_neighbors = min(100, N_rays // 2)
        if n_neighbors > 0:
            idx1 = torch.randint(0, N_rays, (n_neighbors,), device=device)
            idx2 = torch.randint(0, N_rays, (n_neighbors,), device=device)
            
            depth_diff = torch.abs(depth_map[idx1] - depth_map[idx2])
            # Use edge-preserving smoothness (penalize small discontinuities more)
            smoothness_loss = torch.exp(-depth_diff).mean()
            total_loss += smoothness_weight * (1.0 - smoothness_loss)
    
    # Normalize by number of planes found
    if n_planes > 0:
        total_loss = total_loss / n_planes
        
    return total_loss


def manhattan_world_loss(normals: torch.Tensor, weight: float = 1.0,
                        confidence_threshold: float = 0.5) -> torch.Tensor:
    """
    Manhattan-world assumption: encourage normals to align with 3 orthogonal axes.
    
    Args:
        normals: Surface normals [N_rays, N_samples, 3] or [N_rays, 3]
        weight: Loss weight
        confidence_threshold: Minimum confidence for normal prediction
        
    Returns:
        Manhattan world loss
    """
    if normals.dim() == 3:
        # Take the normals with highest confidence (or last sample)
        normals = normals[:, -1, :]  # [N_rays, 3]
    
    device = normals.device
    
    # Define canonical Manhattan directions (world coordinate system)
    manhattan_dirs = torch.tensor([
        [1.0, 0.0, 0.0],  # X-axis (e.g., walls)
        [0.0, 1.0, 0.0],  # Y-axis (e.g., walls)  
        [0.0, 0.0, 1.0],  # Z-axis (e.g., floor/ceiling)
    ], device=device, dtype=normals.dtype)
    
    # Normalize normals
    normals_norm = F.normalize(normals, dim=-1)
    
    # Compute alignment with each Manhattan direction
    alignments = torch.abs(torch.matmul(normals_norm, manhattan_dirs.T))  # [N_rays, 3]
    
    # For each normal, find the best Manhattan direction
    best_alignment, best_idx = torch.max(alignments, dim=-1)  # [N_rays]
    
    # Loss is the deviation from perfect alignment
    manhattan_loss = (1.0 - best_alignment).mean()
    
    return weight * manhattan_loss


def normal_consistency_loss(normals_pred: torch.Tensor, normals_prior: Optional[torch.Tensor] = None,
                          weight: float = 1.0) -> torch.Tensor:
    """
    Normal consistency loss to encourage smooth normal fields.
    
    Args:
        normals_pred: Predicted normals [N_rays, 3]
        normals_prior: Prior normals (if available) [N_rays, 3]
        weight: Loss weight
        
    Returns:
        Normal consistency loss
    """
    device = normals_pred.device
    
    if normals_prior is not None:
        # Cosine similarity loss with prior normals
        normals_pred_norm = F.normalize(normals_pred, dim=-1)
        normals_prior_norm = F.normalize(normals_prior, dim=-1)
        
        cosine_sim = torch.sum(normals_pred_norm * normals_prior_norm, dim=-1)
        consistency_loss = (1.0 - cosine_sim).mean()
        
        return weight * consistency_loss
    
    # If no prior, encourage smoothness between neighboring normals
    N_rays = normals_pred.shape[0]
    if N_rays > 1:
        n_pairs = min(100, N_rays // 2)
        if n_pairs > 0:
            idx1 = torch.randint(0, N_rays, (n_pairs,), device=device)
            idx2 = torch.randint(0, N_rays, (n_pairs,), device=device)
            
            normal1 = F.normalize(normals_pred[idx1], dim=-1)
            normal2 = F.normalize(normals_pred[idx2], dim=-1)
            
            cosine_sim = torch.sum(normal1 * normal2, dim=-1)
            smoothness_loss = (1.0 - cosine_sim).mean()
            
            return weight * smoothness_loss
    
    return torch.tensor(0.0, device=device)


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
    Combine all structural losses for PocketNeRF.
    
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
    
    # Planarity loss  
    if 'planarity' in weights and rays_d is not None:
        planar_loss = planarity_loss(points, normals, depth_pred, rays_d, weights['planarity'])
        loss_dict['planarity'] = planar_loss
        total_loss += planar_loss
    
    # Manhattan world loss
    if normals is not None and 'manhattan' in weights:
        manhattan_loss = manhattan_world_loss(normals, weights['manhattan'])
        loss_dict['manhattan'] = manhattan_loss
        total_loss += manhattan_loss
    
    # Normal consistency loss
    if normals is not None and 'normal_consistency' in weights:
        normal_loss = normal_consistency_loss(normals, weight=weights['normal_consistency'])
        loss_dict['normal_consistency'] = normal_loss  
        total_loss += normal_loss
    
    return total_loss, loss_dict 