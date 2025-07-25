import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
from collections import defaultdict

# Estimates Manhattan coordinate frame from surface normals using clustering
# Using research from: "Surface Normal Clustering for Implicit Representation of Manhattan Scenes"
class ManhattanFrameEstimator:
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.manhattan_frame = None
        self.frame_confidence = 0.0
        
    # Estimate Manhattan frame from surface normals using robust clustering
    def estimate_frame(self, normals: torch.Tensor, confidences: torch.Tensor = None) -> torch.Tensor:
        device = normals.device
        normals = F.normalize(normals, dim=-1)
        
        if confidences is not None:
            # Filter by confidence with more lenient threshold
            mask = confidences > self.confidence_threshold
            if torch.sum(mask) < 20:  
                return torch.eye(3, device=device)
            normals = normals[mask]
        
        if normals.shape[0] < 30:
            return torch.eye(3, device=device)
        
        cluster_centers = self._cluster_normals(normals)
        
        if cluster_centers is not None:
            try:
                U, _, Vt = torch.svd(cluster_centers.T)
                manhattan_frame = U @ Vt
                
                if torch.det(manhattan_frame) < 0:
                    manhattan_frame[:, -1] *= -1
                    
                self.manhattan_frame = manhattan_frame
                return manhattan_frame
            except:
                return torch.eye(3, device=device)
        else:
            return torch.eye(3, device=device)
    
    # Simple k-means clustering for normal directions
    def _cluster_normals(self, normals: torch.Tensor, n_clusters: int = 3) -> Optional[torch.Tensor]:
        device = normals.device
        n_points = normals.shape[0]
        
        if n_points < n_clusters:
            return None
            
        # Initialize cluster centers randomly
        centers = F.normalize(torch.randn(n_clusters, 3, device=device), dim=-1)
        
        # Simple k-means iterations
        for _ in range(10):
            # Assign points to clusters
            similarities = torch.matmul(normals, centers.T)  # [N, 3]
            assignments = torch.argmax(similarities, dim=-1)  # [N]
            
            # Update centers
            new_centers = []
            for k in range(n_clusters):
                mask = assignments == k
                if torch.sum(mask) > 0:
                    center = torch.mean(normals[mask], dim=0)
                    center = F.normalize(center, dim=-1)
                    new_centers.append(center)
                else:
                    new_centers.append(centers[k])
            
            centers = torch.stack(new_centers)
        
        return centers

# Detects semantic planes (floor, walls) for targeted geometric constraints
class SemanticPlaneDetector:
    def __init__(self, depth_threshold: float = 0.1, normal_threshold: float = 0.6):
        self.depth_threshold = depth_threshold
        self.normal_threshold = normal_threshold
        
    def detect_planes(self, depth_map: torch.Tensor, normals: torch.Tensor, 
                     image_coords: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Detect semantic planes from depth and normals with stability improvements.
        
        Args:
            depth_map: Rendered depth [N_rays]
            normals: Surface normals [N_rays, 3]
            image_coords: 2D coordinates [N_rays, 2] (optional)
            
        Returns:
            Dictionary with plane masks and parameters
        """
        device = depth_map.device
        n_rays = depth_map.shape[0]
        
        normals_norm = F.normalize(normals, dim=-1)
        
        # Filter out very small normals (unstable predictions)
        normal_magnitude = torch.norm(normals, dim=-1)
        stable_mask = normal_magnitude > 0.1  
        
        if stable_mask.sum() < 10:  # Not enough stable normals
            return {
                'floor_mask': torch.zeros(n_rays, dtype=torch.bool, device=device),
                'wall_mask': torch.zeros(n_rays, dtype=torch.bool, device=device),
                'wall_clusters': {},
                'n_floor': 0,
                'n_wall': 0
            }
        
        # Apply stability mask
        stable_normals = normals_norm[stable_mask]
        
        # Detect floor (upward facing normals)
        up_vector = torch.tensor([0, 0, 1], device=device, dtype=normals.dtype)
        floor_alignment = torch.abs(torch.sum(stable_normals * up_vector, dim=-1))
        floor_mask_stable = floor_alignment > self.normal_threshold
        
        # Detect walls (horizontal normals)
        horizontal_alignment = torch.abs(stable_normals[:, 2])  # z-component
        wall_mask_stable = horizontal_alignment < (1 - self.normal_threshold)
        
        # Map back to original indices
        stable_indices = torch.where(stable_mask)[0]
        floor_mask = torch.zeros(n_rays, dtype=torch.bool, device=device)
        wall_mask = torch.zeros(n_rays, dtype=torch.bool, device=device)
        
        if floor_mask_stable.sum() > 0:
            floor_indices = stable_indices[floor_mask_stable]
            floor_mask[floor_indices] = True
            
        if wall_mask_stable.sum() > 0:
            wall_indices = stable_indices[wall_mask_stable]
            wall_mask[wall_indices] = True
        
        # Find dominant wall directions (only if enough wall points)
        wall_normals = stable_normals[wall_mask_stable]
        if wall_mask_stable.sum() > 20: 
            wall_clusters = self._cluster_wall_normals(wall_normals)
        else:
            wall_clusters = {}
            
        return {
            'floor_mask': floor_mask,
            'wall_mask': wall_mask,
            'wall_clusters': wall_clusters,
            'n_floor': floor_mask.sum().item(),
            'n_wall': wall_mask.sum().item()
        }
    
    # Cluster wall normals into dominant directions
    def _cluster_wall_normals(self, wall_normals: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = wall_normals.device
        
        # Project to horizontal plane (remove z-component)
        wall_normals_2d = wall_normals[:, :2]
        wall_normals_2d = F.normalize(wall_normals_2d, dim=-1)
        
        # Find two dominant directions (assuming rectangular rooms)
        if wall_normals_2d.shape[0] < 5:
            return {}
            
        # Simple clustering into 2 groups
        # Use the two most separated normals as initial centers
        similarities = torch.matmul(wall_normals_2d, wall_normals_2d.T)
        min_sim_idx = torch.argmin(similarities)
        i, j = min_sim_idx // similarities.shape[1], min_sim_idx % similarities.shape[1]
        
        center1 = wall_normals_2d[i]
        center2 = wall_normals_2d[j]
        
        # Assign to clusters
        sim1 = torch.sum(wall_normals_2d * center1, dim=-1)
        sim2 = torch.sum(wall_normals_2d * center2, dim=-1)
        
        cluster1_mask = sim1 > sim2
        cluster2_mask = ~cluster1_mask
        
        clusters = {}
        if cluster1_mask.sum() > 0:
            clusters['wall_1'] = torch.mean(wall_normals_2d[cluster1_mask], dim=0)
        if cluster2_mask.sum() > 0:
            clusters['wall_2'] = torch.mean(wall_normals_2d[cluster2_mask], dim=0)
            
        return clusters


# ManhattanSDF-style loss with semantic plane constraints
def manhattan_sdf_loss(normals: torch.Tensor, depth_map: torch.Tensor, 
                      manhattan_frame: torch.Tensor, semantic_info: Dict,
                      weight: float = 1.0) -> Tuple[torch.Tensor, Dict]:
    device = normals.device
    normals_norm = F.normalize(normals, dim=-1)
    
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device)
    
    min_points_floor = 50    
    min_points_wall = 30     
    
    # Floor normal constraint (should align with Manhattan up direction)
    if semantic_info['n_floor'] > min_points_floor:
        floor_mask = semantic_info['floor_mask']
        floor_normals = normals_norm[floor_mask]
        
        # Expected floor normal in Manhattan frame
        manhattan_up = manhattan_frame[:, 2]  # Z-axis
        
        # Cosine similarity loss with clamping to prevent extreme values
        floor_alignment = torch.sum(floor_normals * manhattan_up, dim=-1)
        floor_loss = torch.mean(torch.clamp(1.0 - torch.abs(floor_alignment), 0.0, 1.0))
        
        loss_dict['floor'] = floor_loss
        total_loss += floor_loss * 0.5  
    
    # Wall normal constraints (should align with Manhattan horizontal directions)
    if semantic_info['n_wall'] > min_points_wall:
        wall_mask = semantic_info['wall_mask']
        wall_normals = normals_norm[wall_mask]
        
        # Manhattan horizontal directions
        manhattan_x = manhattan_frame[:, 0]
        manhattan_y = manhattan_frame[:, 1]
        
        # Find best alignment with either X or Y axis
        align_x = torch.abs(torch.sum(wall_normals * manhattan_x, dim=-1))
        align_y = torch.abs(torch.sum(wall_normals * manhattan_y, dim=-1))
        
        best_alignment = torch.maximum(align_x, align_y)
        wall_loss = torch.mean(torch.clamp(1.0 - best_alignment, 0.0, 1.0))
        
        loss_dict['wall'] = wall_loss
        total_loss += wall_loss * 0.3  
    
    # General Manhattan alignment for all normals (much lower weight)
    manhattan_dirs = manhattan_frame  # 3x3 matrix
    all_alignments = torch.abs(torch.matmul(normals_norm, manhattan_dirs))  # [N, 3]
    best_alignments = torch.max(all_alignments, dim=-1)[0]  # [N]
    
    # Only penalize normals that are somewhat confident, with stricter threshold
    confidence_mask = best_alignments > 0.5  
    if confidence_mask.sum() > 20:  
        confident_alignments = best_alignments[confidence_mask]
        general_loss = torch.mean(torch.clamp(1.0 - confident_alignments, 0.0, 1.0))
        loss_dict['general'] = general_loss
        total_loss += general_loss * 0.02  
    
    # Clamp total loss to prevent explosion
    total_loss = torch.clamp(total_loss, 0.0, 0.1)
    
    return weight * total_loss, loss_dict

# StructNeRF-style planarity loss with semantic awareness
def structured_planarity_loss(depth_map: torch.Tensor, normals: torch.Tensor,
                            rays_d: torch.Tensor, semantic_info: Dict,
                            weight: float = 1.0, smoothness_scale: float = 0.05) -> torch.Tensor:
    device = depth_map.device
    n_rays = depth_map.shape[0]
    
    if n_rays < 10:
        return torch.tensor(0.0, device=device)
    
    total_loss = torch.tensor(0.0, device=device)
    
    # Floor region smoothness
    if semantic_info['n_floor'] > 5:
        floor_mask = semantic_info['floor_mask']
        floor_indices = torch.where(floor_mask)[0]
        
        if len(floor_indices) > 1:
            # Sample pairs within floor region
            n_pairs = min(100, len(floor_indices) // 2)
            if n_pairs > 0:
                idx = torch.randperm(len(floor_indices))[:n_pairs*2]
                idx1 = floor_indices[idx[:n_pairs]]
                idx2 = floor_indices[idx[n_pairs:2*n_pairs]]
                
                depth_diff = torch.abs(depth_map[idx1] - depth_map[idx2])
                floor_smoothness = torch.mean(depth_diff)
                total_loss += floor_smoothness * 2.0  # Strong smoothness for floor
    
    # Wall region smoothness (within each wall cluster)
    if semantic_info['n_wall'] > 5:
        wall_mask = semantic_info['wall_mask']
        wall_indices = torch.where(wall_mask)[0]
        
        if len(wall_indices) > 1:
            n_pairs = min(100, len(wall_indices) // 2)
            if n_pairs > 0:
                idx = torch.randperm(len(wall_indices))[:n_pairs*2]
                idx1 = wall_indices[idx[:n_pairs]]
                idx2 = wall_indices[idx[n_pairs:2*n_pairs]]
                
                depth_diff = torch.abs(depth_map[idx1] - depth_map[idx2])
                wall_smoothness = torch.mean(depth_diff)
                total_loss += wall_smoothness * 1.5  # Moderate smoothness for walls
    
    # General smoothness for remaining regions (much weaker)
    other_mask = ~(semantic_info['floor_mask'] | semantic_info['wall_mask'])
    if other_mask.sum() > 5:
        other_indices = torch.where(other_mask)[0]
        if len(other_indices) > 1:
            n_pairs = min(50, len(other_indices) // 2)
            if n_pairs > 0:
                idx = torch.randperm(len(other_indices))[:n_pairs*2]
                idx1 = other_indices[idx[:n_pairs]]
                idx2 = other_indices[idx[n_pairs:2*n_pairs]]
                
                depth_diff = torch.abs(depth_map[idx1] - depth_map[idx2])
                other_smoothness = torch.mean(depth_diff)
                total_loss += other_smoothness * 0.1  
    
    return weight * total_loss

# Spatially-aware normal consistency based on actual spatial proximity
def spatial_normal_consistency_loss(normals: torch.Tensor, depth_map: torch.Tensor,
                                  spatial_coords: torch.Tensor = None,
                                  weight: float = 1.0) -> torch.Tensor:
    device = normals.device
    n_rays = normals.shape[0]
    
    if n_rays < 10:
        return torch.tensor(0.0, device=device)
    
    normals_norm = F.normalize(normals, dim=-1)
    
    # If spatial coordinates available, use spatial proximity
    if spatial_coords is not None:
        # Find spatially close pairs
        n_pairs = min(200, n_rays // 2)
        idx1 = torch.randint(0, n_rays, (n_pairs,), device=device)
        
        # For each point, find its closest spatial neighbor
        distances = torch.cdist(spatial_coords[idx1], spatial_coords)  # [n_pairs, n_rays]
        distances[torch.arange(n_pairs), idx1] = float('inf')  # Exclude self
        idx2 = torch.argmin(distances, dim=-1)
        
        # Weight by spatial proximity and depth similarity
        spatial_dist = distances[torch.arange(n_pairs), idx2]
        depth_similarity = torch.exp(-torch.abs(depth_map[idx1] - depth_map[idx2]))
        spatial_weight = torch.exp(-spatial_dist * 0.1)  # Exponential decay with distance
        
        # Normal consistency weighted by spatial and depth proximity
        normal1 = normals_norm[idx1]
        normal2 = normals_norm[idx2]
        cosine_sim = torch.sum(normal1 * normal2, dim=-1)
        
        weights = spatial_weight * depth_similarity
        consistency_loss = torch.mean(weights * (1.0 - cosine_sim))
        
    else:
        # Fallback: use sequential neighbors (assuming some spatial ordering)
        n_pairs = min(100, n_rays - 1)
        idx1 = torch.randint(0, n_rays - 1, (n_pairs,), device=device)
        idx2 = idx1 + 1
        
        # Weight by depth similarity
        depth_similarity = torch.exp(-torch.abs(depth_map[idx1] - depth_map[idx2]))
        
        normal1 = normals_norm[idx1]
        normal2 = normals_norm[idx2]
        cosine_sim = torch.sum(normal1 * normal2, dim=-1)
        
        consistency_loss = torch.mean(depth_similarity * (1.0 - cosine_sim))
    
    return weight * consistency_loss

# Advanced structural losses combining ManhattanSDF and StructNeRF approaches
def combine_structural_losses_v2(depth_pred: torch.Tensor,
                               normals: torch.Tensor,
                               rays_d: torch.Tensor,
                               spatial_coords: torch.Tensor = None,
                               weights: Dict[str, float] = None,
                               manhattan_frame_estimator: ManhattanFrameEstimator = None,
                               semantic_detector: SemanticPlaneDetector = None) -> Tuple[torch.Tensor, Dict]:
    if weights is None:
        weights = {
            'manhattan': 1.0,
            'planarity': 1.0,
            'normal_consistency': 0.5
        }
    
    device = depth_pred.device
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=device)
    
    if normals is None:
        print("⚠️  Structural priors: normals is None, skipping normal-based losses")
        return total_loss, {'error': 'normals_none'}
    
    if normals.dim() != 2 or normals.shape[-1] != 3:
        print(f"⚠️  Structural priors: invalid normals shape {normals.shape}, expected [N, 3]")
        return total_loss, {'error': f'invalid_normals_shape_{normals.shape}'}
    
    if normals.shape[0] == 0:
        print("⚠️  Structural priors: empty normals tensor, skipping")
        return total_loss, {'error': 'empty_normals'}
    
    if depth_pred.shape[0] != normals.shape[0]:
        print(f"⚠️  Structural priors: depth-normal size mismatch {depth_pred.shape[0]} vs {normals.shape[0]}")
        return total_loss, {'error': f'size_mismatch_{depth_pred.shape[0]}_{normals.shape[0]}'}
    
    if manhattan_frame_estimator is None:
        manhattan_frame_estimator = ManhattanFrameEstimator(confidence_threshold=0.4)  
    if semantic_detector is None:
        semantic_detector = SemanticPlaneDetector(normal_threshold=0.5)  
    
    try:
        semantic_info = semantic_detector.detect_planes(depth_pred, normals, spatial_coords)
        
        normal_confidences = torch.norm(normals, dim=-1)  # Use normal magnitude as confidence
        manhattan_frame = manhattan_frame_estimator.estimate_frame(normals, normal_confidences)
        
        # Manhattan loss with semantic awareness
        if 'manhattan' in weights and normals is not None:
            manhattan_loss, manhattan_dict = manhattan_sdf_loss(
                normals, depth_pred, manhattan_frame, semantic_info, weights['manhattan']
            )
            loss_dict.update({f'manhattan_{k}': v for k, v in manhattan_dict.items()})
            total_loss += manhattan_loss
        
        # Structured planarity loss
        if 'planarity' in weights:
            planarity_loss = structured_planarity_loss(
                depth_pred, normals, rays_d, semantic_info, weights['planarity']
            )
            loss_dict['planarity'] = planarity_loss
            total_loss += planarity_loss
        
        # Spatial normal consistency
        if 'normal_consistency' in weights and normals is not None:
            consistency_loss = spatial_normal_consistency_loss(
                normals, depth_pred, spatial_coords, weights['normal_consistency']
            )
            loss_dict['normal_consistency'] = consistency_loss
            total_loss += consistency_loss
        
        # Add semantic info to loss dict for logging
        loss_dict['semantic_floor_count'] = semantic_info['n_floor']
        loss_dict['semantic_wall_count'] = semantic_info['n_wall']
        
    except Exception as e:
        print(f"⚠️  Structural priors computation failed: {e}")
        return torch.tensor(0.0, device=device), {'error': str(e)}
    
    return total_loss, loss_dict 