from abc import ABC, abstractmethod
import numpy as np
import torch
from ..flex.flex_base import FlexBase
from .mask_base import MaskBase
from ...tooltips import apply_tooltips

@apply_tooltips
class FlexMaskBase(FlexBase, MaskBase):
    """Base class for Flex-enabled mask operations that combines FlexBase feature modulation with MaskBase operations."""
    
    @classmethod
    def INPUT_TYPES(cls):
        # Get input types from FlexBase
        base_inputs = super().INPUT_TYPES()
        
        # Get MaskBase inputs
        mask_inputs = MaskBase.INPUT_TYPES()
        
        # First rename MaskBase's strength to mask_strength so as to  not conflict with FlexBase's strength
        mask_inputs["required"]["mask_strength"] = mask_inputs["required"].pop("strength")
        
        # Update the base inputs with mask inputs
        base_inputs["required"].update(mask_inputs["required"])
        
        
        # Update optional inputs
        if "optional" in mask_inputs:
            if "optional" not in base_inputs:
                base_inputs["optional"] = {}
            base_inputs["optional"].update(mask_inputs["optional"])
        
        return base_inputs

    CATEGORY = "RyanOnTheInside/FlexFeatures/Targets/Masks"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply_effect"

    def __init__(self):
        # Initialize both parent classes
        FlexBase.__init__(self)
        MaskBase.__init__(self)
        self.parameter_scheduler = None

    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        return ["None"]


    #TODO update this to make the contract more clear (feature_mode, feature_param, etc)
    @abstractmethod
    def apply_effect_internal(self, mask: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect with processed parameters.
        
        Args:
            mask: The input mask to process
            **kwargs: All parameters needed for the effect, already processed by the base class
            
        Returns:
            The processed mask
        """
        pass

    def process_mask(self, mask: np.ndarray, strength: float, **kwargs) -> np.ndarray:
        """Implementation of MaskBase's abstract process_mask method."""
        # This satisfies MaskBase's interface by delegating to the flex system
        return self.apply_effect(
            masks=torch.from_numpy(mask).unsqueeze(0),  # Add batch dimension
            strength=strength,
            **kwargs
        )[0]  # Remove batch dimension

    def process_below_threshold(self, mask, **kwargs):
        """Default behavior for when feature value is below threshold: return mask unchanged."""
        return mask

    def apply_effect(self, masks, opt_feature=None, strength=1.0, feature_threshold=0.0, 
                    mask_strength=1.0, invert=False, subtract_original=0.0, 
                    grow_with_blur=0.0, feature_param=None, feature_mode="relative", **kwargs):
        """Main entry point for the Flex system."""
        if opt_feature is not None:
            num_frames = opt_feature.frame_count
        else:
            # Start with number of input frames
            num_frames = masks.shape[0]
            # Check all parameters for lists/arrays that might be longer
            for value in kwargs.values():
                if isinstance(value, (list, tuple, np.ndarray)):
                    num_frames = max(num_frames, len(value))

        original_masks = masks.clone()

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        result = []
        for i in range(num_frames):
            mask = masks[i % masks.shape[0]].numpy()
            
            # Get feature value
            feature_value = self.get_feature_value(i, opt_feature)
            
            # Process parameters using FlexBase functionality
            processed_kwargs = self.process_parameters(
                frame_index=i,
                feature_value=feature_value,
                feature_threshold=feature_threshold,
                strength=strength,
                mask_strength=mask_strength,
                subtract_original=subtract_original,
                grow_with_blur=grow_with_blur,
                feature_param=feature_param,
                feature_mode=feature_mode,
                **kwargs
            )

            # Determine if effect should be applied based on feature value and threshold
            if feature_value is not None and feature_value >= processed_kwargs['feature_threshold']:
                processed_mask = self.apply_effect_internal(
                    mask,
                    **processed_kwargs
                )
            else:
                processed_mask = self.process_below_threshold(
                    mask,
                    **processed_kwargs
                )

            # Apply mask operations using modulo for original mask indexing
            frame_result = self.apply_mask_operation(
                processed_mask[np.newaxis, ...],
                original_masks[i % masks.shape[0]:i % masks.shape[0]+1],
                processed_kwargs['mask_strength'],
                invert,
                processed_kwargs['subtract_original'],
                processed_kwargs['grow_with_blur'],
                progress_callback=self.update_progress
            )
            result.append(frame_result)

        self.end_progress()
        return (torch.cat(result, dim=0),)
    
    def main_function(self, masks, opt_feature=None, strength=1.0, feature_threshold=0.0, 
                     mask_strength=1.0, invert=False, subtract_original=0.0, 
                     grow_with_blur=0.0, **kwargs):
        """Implementation of MaskBase's abstract main_function."""
        return self.apply_effect(
            masks=masks,
            opt_feature=opt_feature,
            strength=strength,
            feature_threshold=feature_threshold,
            mask_strength=mask_strength,
            invert=invert,
            subtract_original=subtract_original,
            grow_with_blur=grow_with_blur,
            **kwargs
        )
