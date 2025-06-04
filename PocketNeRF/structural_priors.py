import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


def depth_prior_loss(depth_pred: torch.Tensor, depth_prior: torch.Tensor, 
                    weight: float = 1.0, use_ranking: bool = True) -> torch.Tensor:
    """
    Depth prior loss with stronger regularization for few-shot scenarios.
    """
    if use_ranking:
        # Increase number of pairs for better coverage
        n_pairs = min(5000, depth_pred.shape[0] * 10)  # Much more pairs
        if n_pairs > 0:
            idx1 = torch.randint(0, depth_pred.shape[0], (n_pairs,), device=depth_pred.device)
            idx2 = torch.randint(0, depth_pred.shape[0], (n_pairs,), device=depth_pred.device)
            
            # Only compare pairs that are spatially close (more relevant comparisons)
            valid_pairs = (idx1 - idx2).abs() < depth_pred.shape[0] // 4
            idx1 = idx1[valid_pairs]
            idx2 = idx2[valid_pairs]
            
            if len(idx1) > 0:
                depth_diff_pred = depth_pred[idx1] - depth_pred[idx2]
                depth_diff_prior = depth_prior[idx1] - depth_prior[idx2]
                
                # Stronger hinge loss with margin
                margin = 0.1
                loss = F.relu(margin - depth_diff_pred * depth_diff_prior.sign()).mean()
                return weight * loss
    else:
        # Robust loss function to handle outliers better
        depth_pred_norm = (depth_pred - depth_pred.median()) / (depth_pred.std() + 1e-8)
        depth_prior_norm = (depth_prior - depth_prior.median()) / (depth_prior.std() + 1e-8)
        
        # Huber loss is more robust to outliers than MSE
        return weight * F.smooth_l1_loss(depth_pred_norm, depth_prior_norm)
    
    return torch.tensor(0.0, device=depth_pred.device)


def planarity_loss(points: torch.Tensor, normals: torch.Tensor, 
                  depth_map: torch.Tensor, rays_d: torch.Tensor,
                  weight: float = 1.0, plane_threshold: float = 0.05,
                  smoothness_weight: float = 0.5) -> torch.Tensor:
    """
    Stronger planarity constraints with depth smoothness and normal alignment.
    """
    device = points.device
    N_rays = depth_map.shape[0]
    
    if N_rays < 3:
        return torch.tensor(0.0, device=device)
    
    # Convert depth map to 3D points on surface
    surface_points = rays_d * depth_map.unsqueeze(-1)
    
    total_loss = torch.tensor(0.0, device=device)
    
    # 1. Local planarity constraint - check small neighborhoods
    patch_size = int(np.sqrt(N_rays))
    if patch_size > 10:  # Only if we have enough rays
        # Reshape to approximate image grid
        try:
            depth_grid = depth_map[:patch_size**2].view(patch_size, patch_size)
            
            # Compute local depth variance in small windows
            kernel_size = 3
            unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=1, padding=1)
            patches = unfold(depth_grid.unsqueeze(0).unsqueeze(0))  # [1, k*k, n_patches]
            
            # Penalize high variance in local patches
            patch_vars = patches.var(dim=1)
            planarity_loss = patch_vars.mean()
            total_loss += weight * 2.0 * planarity_loss  # Stronger weight
        except:
            pass
    
    # 2. Global plane fitting with RANSAC-style approach
    n_trials = min(100, N_rays // 10)
    plane_losses = []
    
    for _ in range(n_trials):
        if N_rays >= 3:
            indices = torch.randperm(N_rays, device=device)[:3]
            plane_points = surface_points[indices]
            
            # Fit plane
            v1 = plane_points[1] - plane_points[0]
            v2 = plane_points[2] - plane_points[0]
            normal = torch.linalg.cross(v1, v2)
            normal_norm = torch.norm(normal)
            
            if normal_norm > 1e-6:
                normal = normal / normal_norm
                plane_origin = plane_points[0]
                
                # Compute distances to plane
                distances = torch.abs(torch.sum((surface_points - plane_origin) * normal, dim=-1))
                
                # Adaptive threshold based on scene scale
                adaptive_threshold = plane_threshold * depth_map.mean()
                on_plane_mask = distances < adaptive_threshold
                n_on_plane = torch.sum(on_plane_mask)
                
                # If enough points are coplanar, this might be a real plane
                if n_on_plane >= N_rays * 0.1:  # At least 10% of points
                    # Strong penalty for deviations
                    plane_loss = distances[on_plane_mask].mean()
                    plane_losses.append(plane_loss)
                    
                    # Also encourage normal consistency if we have normals
                    if normals is not None and normals.shape[0] == N_rays:
                        normal_alignment = 1.0 - torch.abs(torch.sum(normals[on_plane_mask] * normal, dim=-1)).mean()
                        total_loss += weight * 0.5 * normal_alignment
    
    if plane_losses:
        # Use the best plane (lowest loss) as strong guidance
        best_plane_loss = torch.stack(plane_losses).min()
        total_loss += weight * 3.0 * best_plane_loss
    
    # 3. Depth smoothness with edge-aware weighting
    n_neighbors = min(1000, N_rays)
    if n_neighbors > 0:
        idx1 = torch.randint(0, N_rays, (n_neighbors,), device=device)
        idx2 = torch.randint(0, N_rays, (n_neighbors,), device=device)
        
        # Spatial distance weight (closer pixels should have similar depth)
        ray_similarity = F.cosine_similarity(rays_d[idx1], rays_d[idx2], dim=-1).abs()
        
        # Depth difference weighted by spatial proximity
        depth_diff = torch.abs(depth_map[idx1] - depth_map[idx2])
        weighted_diff = depth_diff * ray_similarity
        
        # Total variation regularization
        tv_loss = weighted_diff.mean()
        total_loss += smoothness_weight * tv_loss
        
    return total_loss


def manhattan_world_loss(normals: torch.Tensor, weight: float = 1.0,
                        confidence_threshold: float = 0.5) -> torch.Tensor:
    """
    Stronger Manhattan-world assumption with adaptive axis detection.
    """
    if normals.dim() == 3:
        normals = normals[:, -1, :]
    
    device = normals.device
    
    # Ensure normals are valid
    normal_norms = torch.norm(normals, dim=-1)
    valid_mask = normal_norms > 1e-6
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)
    
    valid_normals = normals[valid_mask]
    valid_normals = F.normalize(valid_normals, dim=-1)
    
    # Define canonical Manhattan directions
    manhattan_dirs = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], device=device, dtype=normals.dtype)
    
    # Compute alignment with each Manhattan direction
    alignments = torch.abs(torch.matmul(valid_normals, manhattan_dirs.T))
    
    # For each normal, find the best Manhattan direction
    best_alignment, best_idx = torch.max(alignments, dim=-1)
    
    # Stronger penalty for deviation from perfect alignment
    # Use squared loss for stronger gradient when far from Manhattan
    manhattan_loss = torch.pow(1.0 - best_alignment, 2).mean()
    
    # Additional loss to encourage diversity of Manhattan directions
    # (prevent all normals from aligning to same axis)
    if len(valid_normals) > 10:
        direction_counts = torch.bincount(best_idx, minlength=3).float()
        direction_probs = direction_counts / direction_counts.sum()
        # Entropy regularization
        entropy = -(direction_probs * torch.log(direction_probs + 1e-8)).sum()
        diversity_loss = torch.relu(1.5 - entropy)  # Encourage entropy > 1.5
        manhattan_loss += 0.1 * diversity_loss
    
    return weight * manhattan_loss


def normal_consistency_loss(normals_pred: torch.Tensor, normals_prior: Optional[torch.Tensor] = None,
                          weight: float = 1.0, spatial_weight: float = 2.0) -> torch.Tensor:
    """
    Stronger normal consistency with spatial coherence.
    """
    device = normals_pred.device
    
    if normals_prior is not None:
        normals_pred_norm = F.normalize(normals_pred, dim=-1)
        normals_prior_norm = F.normalize(normals_prior, dim=-1)
        
        # Angular loss (more sensitive than cosine similarity)
        dot_product = torch.sum(normals_pred_norm * normals_prior_norm, dim=-1).clamp(-1, 1)
        angle_loss = torch.acos(dot_product).mean()
        
        return weight * angle_loss
    
    # Spatial smoothness for neighboring normals
    N_rays = normals_pred.shape[0]
    if N_rays > 1:
        # Sample many more pairs for better coverage
        n_pairs = min(2000, N_rays * 5)
        if n_pairs > 0:
            idx1 = torch.randint(0, N_rays, (n_pairs,), device=device)
            idx2 = torch.randint(0, N_rays, (n_pairs,), device=device)
            
            # Only compare nearby normals (spatial coherence)
            spatial_dist = (idx1 - idx2).abs().float() / N_rays
            spatial_weight_mask = torch.exp(-spatial_dist * 10.0)  # Exponential decay
            
            normal1 = F.normalize(normals_pred[idx1], dim=-1)
            normal2 = F.normalize(normals_pred[idx2], dim=-1)
            
            # Angular difference weighted by spatial proximity
            dot_product = torch.sum(normal1 * normal2, dim=-1).clamp(-1, 1)
            angle_diff = torch.acos(dot_product)
            weighted_angle_diff = angle_diff * spatial_weight_mask
            
            smoothness_loss = weighted_angle_diff.mean()
            
            return weight * spatial_weight * smoothness_loss
    
    return torch.tensor(0.0, device=device)


def edge_aware_smoothness_loss(depth_map: torch.Tensor, rgb_map: torch.Tensor, 
                              weight: float = 1.0) -> torch.Tensor:
    """
    Edge-aware depth smoothness loss.
    Encourages smooth depth except at color edges.
    """
    device = depth_map.device
    N_rays = depth_map.shape[0]
    
    if N_rays < 4:
        return torch.tensor(0.0, device=device)
    
    # Sample neighboring pairs
    n_pairs = min(1000, N_rays)
    idx1 = torch.randint(0, N_rays, (n_pairs,), device=device)
    idx2 = torch.randint(0, N_rays, (n_pairs,), device=device)
    
    # Compute RGB difference (edge strength)
    rgb_diff = torch.norm(rgb_map[idx1] - rgb_map[idx2], dim=-1)
    edge_weight = torch.exp(-rgb_diff * 10.0)  # Low weight where color changes
    
    # Compute depth difference
    depth_diff = torch.abs(depth_map[idx1] - depth_map[idx2])
    
    # Weighted smoothness loss
    weighted_depth_diff = depth_diff * edge_weight
    
    return weight * weighted_depth_diff.mean()


def combine_structural_losses(depth_pred: torch.Tensor, 
                            points: torch.Tensor,
                            normals: Optional[torch.Tensor] = None,
                            rays_d: torch.Tensor = None,
                            depth_prior: Optional[torch.Tensor] = None,
                            rgb_pred: Optional[torch.Tensor] = None,
                            height: int = None, width: int = None,
                            weights: dict = None) -> Tuple[torch.Tensor, dict]:
    """
    Combine all structural losses with adaptive weighting for few-shot scenarios.
    """
    if weights is None:
        weights = {
            'depth_prior': 1.0,
            'planarity': 2.0,      # Increased
            'manhattan': 1.5,       # Increased
            'normal_consistency': 1.0,
            'edge_smoothness': 0.5
        }
    
    device = depth_pred.device
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device)
    
    # Depth prior loss
    if depth_prior is not None and 'depth_prior' in weights:
        depth_loss = depth_prior_loss(depth_pred, depth_prior, weights['depth_prior'])
        loss_dict['depth_prior'] = depth_loss
        total_loss += depth_loss
    
    # Planarity loss with stronger weight
    if 'planarity' in weights and rays_d is not None:
        planar_loss = planarity_loss(points, normals, depth_pred, rays_d, 
                                   weights['planarity'], smoothness_weight=1.0)
        loss_dict['planarity'] = planar_loss
        total_loss += planar_loss
    
    # Manhattan world loss - ensure it actually contributes
    if normals is not None and 'manhattan' in weights:
        # Add small noise to prevent all normals being identical
        normals_noisy = normals + torch.randn_like(normals) * 0.01
        manhattan_loss = manhattan_world_loss(normals_noisy, weights['manhattan'])
        loss_dict['manhattan'] = manhattan_loss
        total_loss += manhattan_loss
    
    # Normal consistency loss with spatial coherence
    if normals is not None and 'normal_consistency' in weights:
        normal_loss = normal_consistency_loss(normals, weight=weights['normal_consistency'], 
                                            spatial_weight=2.0)
        loss_dict['normal_consistency'] = normal_loss
        total_loss += normal_loss
    
    # Edge-aware smoothness
    if rgb_pred is not None and 'edge_smoothness' in weights:
        smoothness_loss = edge_aware_smoothness_loss(depth_pred, rgb_pred, 
                                                    weights['edge_smoothness'])
        loss_dict['edge_smoothness'] = smoothness_loss
        total_loss += smoothness_loss
    
    return total_loss, loss_dict