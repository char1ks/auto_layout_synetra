"""SAM Predictor module for SearchDet pipeline."""

import torch
import numpy as np
from typing import Optional, Tuple, List


class SAMPredictor:
    """SAM predictor wrapper for mask generation."""
    
    def __init__(self, sam_model=None):
        """Initialize SAM predictor.
        
        Args:
            sam_model: SAM model instance
        """
        self.sam_model = sam_model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def set_image(self, image: np.ndarray) -> None:
        """Set image for prediction.
        
        Args:
            image: Input image as numpy array (H, W, 3)
        """
        if self.sam_model is not None:
            self.sam_model.set_image(image)
    
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict masks using SAM.
        
        Args:
            point_coords: Point coordinates for prompting
            point_labels: Point labels (1 for positive, 0 for negative)
            box: Bounding box prompt
            mask_input: Mask input for refinement
            multimask_output: Whether to return multiple masks
            return_logits: Whether to return logits
            
        Returns:
            Tuple of (masks, scores, logits)
        """
        if self.sam_model is None:
            # Return dummy masks if no SAM model
            return np.array([]), np.array([]), np.array([])
            
        return self.sam_model.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits
        )
    
    def reset_image(self) -> None:
        """Reset the currently set image."""
        if hasattr(self.sam_model, 'reset_image'):
            self.sam_model.reset_image()