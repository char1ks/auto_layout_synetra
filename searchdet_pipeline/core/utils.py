"""Utility functions for SearchDet pipeline."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union


def get_image_size(image: Union[np.ndarray, torch.Tensor]) -> Tuple[int, int]:
    """Get image size (height, width).
    
    Args:
        image: Input image as numpy array or torch tensor
        
    Returns:
        Tuple of (height, width)
    """
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            return image.shape[:2]  # (H, W)
        elif len(image.shape) == 2:
            return image.shape  # (H, W)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 4:  # (B, C, H, W)
            return image.shape[2:4]
        elif len(image.shape) == 3:  # (C, H, W)
            return image.shape[1:3]
        elif len(image.shape) == 2:  # (H, W)
            return image.shape
        else:
            raise ValueError(f"Unsupported tensor shape: {image.shape}")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def get_feature_map_size(feature_map: torch.Tensor) -> Tuple[int, int]:
    """Get feature map size (height, width).
    
    Args:
        feature_map: Feature map tensor
        
    Returns:
        Tuple of (height, width)
    """
    if len(feature_map.shape) == 4:  # (B, C, H, W)
        return feature_map.shape[2:4]
    elif len(feature_map.shape) == 3:  # (C, H, W)
        return feature_map.shape[1:3]
    elif len(feature_map.shape) == 2:  # (H, W)
        return feature_map.shape
    else:
        raise ValueError(f"Unsupported feature map shape: {feature_map.shape}")


def upsample_feature_map(
    feature_map: torch.Tensor, 
    target_size: Tuple[int, int], 
    mode: str = 'bilinear',
    align_corners: bool = False
) -> torch.Tensor:
    """Upsample feature map to target size.
    
    Args:
        feature_map: Input feature map tensor
        target_size: Target size (height, width)
        mode: Interpolation mode ('bilinear', 'nearest', etc.)
        align_corners: Whether to align corners in interpolation
        
    Returns:
        Upsampled feature map
    """
    if len(feature_map.shape) == 2:  # (H, W)
        feature_map = feature_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        squeeze_output = True
    elif len(feature_map.shape) == 3:  # (C, H, W)
        feature_map = feature_map.unsqueeze(0)  # (1, C, H, W)
        squeeze_output = False
    elif len(feature_map.shape) == 4:  # (B, C, H, W)
        squeeze_output = False
    else:
        raise ValueError(f"Unsupported feature map shape: {feature_map.shape}")
    
    # Perform upsampling
    upsampled = F.interpolate(
        feature_map, 
        size=target_size, 
        mode=mode, 
        align_corners=align_corners if mode == 'bilinear' else None
    )
    
    # Restore original dimensions if needed
    if squeeze_output:
        if len(feature_map.shape) == 2:
            upsampled = upsampled.squeeze(0).squeeze(0)  # (H, W)
        elif len(feature_map.shape) == 3:
            upsampled = upsampled.squeeze(0)  # (C, H, W)
    
    return upsampled


def normalize_tensor(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Normalize tensor along specified dimension.
    
    Args:
        tensor: Input tensor
        dim: Dimension to normalize along
        
    Returns:
        Normalized tensor
    """
    return F.normalize(tensor, p=2, dim=dim)


def compute_cosine_similarity(
    tensor1: torch.Tensor, 
    tensor2: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity between two tensors.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        
    Returns:
        Cosine similarity tensor
    """
    return F.cosine_similarity(tensor1, tensor2, dim=-1)


def batch_process(
    data: list, 
    batch_size: int, 
    process_fn: callable
) -> list:
    """Process data in batches.
    
    Args:
        data: List of data items
        batch_size: Size of each batch
        process_fn: Function to process each batch
        
    Returns:
        List of processed results
    """
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_result = process_fn(batch)
        results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
    return results