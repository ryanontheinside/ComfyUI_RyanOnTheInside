from abc import ABC, abstractmethod
import numpy as np
import torch
from ..flex.flex_base import FlexBase
from .mask_base import MaskBase

class FlexMaskBase(FlexBase, MaskBase):
    """Base class for Flex-enabled mask operations that combines FlexBase feature modulation with MaskBase operations."""
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get input types from both parent classes
        flex_inputs = FlexBase.INPUT_TYPES()
        mask_inputs = MaskBase.INPUT_TYPES()
        
        # Get the required inputs
        required = {
            **flex_inputs["required"],
            **mask_inputs["required"]
        }
        
        # Rename the mask strength parameter to avoid conflict with flex strength
        required["mask_strength"] = required.pop("strength")
        
        # Add feature-specific inputs
        required.update({
            "feature_param": (cls.get_modifiable_params(),),
            "feature_mode": (["relative", "absolute"], {"default": "relative"}),
        })
        
        # Merge optional inputs
        optional = {
            **flex_inputs.get("optional", {}),
            **mask_inputs.get("optional", {})
        }
        
        return {
            "required": required,
            "optional": optional
        }

    CATEGORY = "RyanOnTheInside/FlexMasks"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "main_function"

    def __init__(self):
        # Initialize both parent classes
        FlexBase.__init__(self)
        MaskBase.__init__(self)

    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        return ["None"]

    @abstractmethod
    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, **kwargs) -> np.ndarray:
        """Process a single mask with feature modulation.
        
        Args:
            mask: Input mask to process
            feature_value: Current feature value for modulation
            strength: Strength of the feature modulation
            **kwargs: Additional parameters
        
        Returns:
            Processed mask as numpy array
        """
        pass

    def apply_mask_operation(self, masks, feature, feature_pipe, strength, feature_threshold, mask_strength, invert, subtract_original, grow_with_blur, **kwargs):
        """Apply mask operation with feature modulation.
        
        This method combines the FlexBase feature modulation pipeline with MaskBase operations.
        """
        num_frames = feature_pipe.frame_count
        original_masks = masks.clone()

        self.start_progress(num_frames, desc="Applying flex mask operation")

        result = []
        for i in range(num_frames):
            kwargs['frame_index'] = i
            mask = masks[i].numpy()
            feature_value = feature.get_value_at_frame(i)
            
            if feature_value >= feature_threshold:
                processed_mask = self.process_mask(mask, feature_value, strength, **kwargs)
            else:
                if hasattr(self, 'process_mask_below_threshold'):
                    processed_mask = self.process_mask_below_threshold(mask, feature_value, strength, **kwargs)
                else:
                    processed_mask = mask

            result.append(processed_mask)
            self.update_progress()

        self.end_progress()

        processed_masks = torch.from_numpy(np.stack(result)).float()
        # Use mask_strength instead of strength for the final mask operation
        return super().apply_mask_operation(processed_masks, original_masks, mask_strength, invert, subtract_original, grow_with_blur, **kwargs)

    @abstractmethod
    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, mask_strength, invert, subtract_original, grow_with_blur, **kwargs):
        """Main entry point for the node.
        
        This method should be implemented by subclasses to define their specific behavior.
        """
        pass