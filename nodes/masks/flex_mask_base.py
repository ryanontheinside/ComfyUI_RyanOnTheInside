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
        # Get input types from FlexBase (which already has list_ok added)
        base_inputs = super().INPUT_TYPES()
        
        # Get MaskBase inputs
        mask_inputs = MaskBase.INPUT_TYPES()
        
        # First rename MaskBase's strength to mask_strength
        mask_inputs["required"]["mask_strength"] = mask_inputs["required"].pop("strength")
        
        # Update the base inputs (preserving list_ok decorators)
        base_inputs["required"].update(mask_inputs["required"])
        
        # Add feature-specific inputs
        base_inputs["required"].update({
            "feature_param": (cls.get_modifiable_params(),),
            "feature_mode": (["relative", "absolute"], {"default": "relative"}),
        })
        
        # Update optional inputs
        if "optional" in mask_inputs:
            if "optional" not in base_inputs:
                base_inputs["optional"] = {}
            base_inputs["optional"].update(mask_inputs["optional"])
        
        return base_inputs

    CATEGORY = "RyanOnTheInside/FlexMasks"
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

    def modulate_parameter_value(self, param_name: str, param_value: float | list | tuple | np.ndarray,
                               feature_value: float, strength: float, mode: str,
                               frame_index: int = 0) -> float:
        """Helper method to consistently handle parameter modulation across all child classes."""
        # Handle array-like parameters
        if isinstance(param_value, (list, tuple, np.ndarray)):
            try:
                base_value = float(param_value[frame_index])
            except (IndexError, TypeError):
                base_value = float(param_value[0])
        else:
            base_value = float(param_value)
        
        # Apply modulation
        return self.modulate_param(param_name, base_value, feature_value, strength, mode)

    def get_feature_value(self, feature, frame_index, default=0.5):
        """Get feature value with consistent handling"""
        if feature is None:
            return default
        try:
            return feature.get_value_at_frame(frame_index)
        except (AttributeError, IndexError):
            return default

    def process_parameters_for_frame(self, frame_index: int, feature_value: float, 
                                   strength: float, **kwargs) -> dict:
        """Process parameters for a single frame, handling both scheduling and modulation."""
        # Process parameters using FlexBase's functionality
        processed_kwargs = self.process_parameters(
            frame_index=frame_index,
            feature_value=feature_value,
            strength=strength,
            **kwargs
        )
        
        # Remove parameters that shouldn't be passed to child classes
        processed_kwargs.pop('feature_param', None)
        processed_kwargs.pop('feature_mode', None)
        processed_kwargs.pop('frame_index', None)
        
        return processed_kwargs

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, **kwargs) -> np.ndarray:
        """Process a single mask with feature modulation and parameter scheduling."""
        # Get the processed parameters for this frame
        processed_kwargs = {}
        feature_param = kwargs.pop('feature_param', None)
        feature_mode = kwargs.pop('feature_mode', None)
        
        # Copy all other parameters
        processed_kwargs.update(kwargs)
        
        # If this parameter is being modulated by a feature, handle it
        if feature_param and feature_param in processed_kwargs and feature_param != "None":
            base_value = float(processed_kwargs[feature_param])
            processed_kwargs[feature_param] = self.modulate_param(
                feature_param, 
                base_value, 
                feature_value, 
                strength, 
                feature_mode
            )
        
        return self.apply_effect_internal(mask, feature_value, strength, **processed_kwargs)

    def apply_effect_internal(self, mask: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect with processed parameters."""
        return self.process_mask(mask, **kwargs)

    def apply_effect(self, masks, opt_feature=None, strength=1.0, feature_threshold=0.0, 
                    mask_strength=1.0, invert=False, subtract_original=0.0, 
                    grow_with_blur=0.0, **kwargs):
        """Main entry point for the Flex system."""
        num_frames = masks.shape[0]
        original_masks = masks.clone()

        #NOTE MaskBase will handle progress bar, annoying yes
        # self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        # Initialize parameter scheduler with all parameters including MaskBase parameters
        if self.parameter_scheduler is None:
            all_params = {
                'mask_strength': mask_strength,
                'subtract_original': subtract_original,
                'grow_with_blur': grow_with_blur,
                **kwargs
            }
            self.initialize_scheduler(num_frames, **all_params)

        result = []
        for i in range(num_frames):
            mask = masks[i].numpy()
            
            # Get feature value and determine if effect should be applied
            feature_value = self.get_feature_value(opt_feature, i)
            apply_effect = feature_value >= feature_threshold if opt_feature is not None else True

            # Process frame-specific parameters including MaskBase parameters
            frame_kwargs = {}
            all_params = {
                'mask_strength': mask_strength,
                'subtract_original': subtract_original,
                'grow_with_blur': grow_with_blur,
                **kwargs
            }
            
            for key, value in all_params.items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    try:
                        frame_val = value[i]
                    except (IndexError, TypeError):
                        frame_val = value[0]
                    
                    # Preserve numeric type
                    if isinstance(frame_val, (int, np.integer)):
                        frame_kwargs[key] = int(frame_val)
                    else:
                        frame_kwargs[key] = float(frame_val)
                else:
                    frame_kwargs[key] = value

            # Extract MaskBase parameters
            current_mask_strength = frame_kwargs.pop('mask_strength')
            current_subtract = frame_kwargs.pop('subtract_original')
            current_blur = frame_kwargs.pop('grow_with_blur')

            # Process frame with frame-specific parameters
            if apply_effect:
                processed_mask = self.process_mask(
                    mask,
                    feature_value=feature_value,
                    strength=strength,
                    **frame_kwargs
                )
            else:
                if hasattr(self, 'process_mask_below_threshold'):
                    processed_mask = self.process_mask_below_threshold(
                        mask,
                        feature_value=feature_value,
                        strength=strength,
                        **frame_kwargs
                    )
                else:
                    processed_mask = mask

            # Apply mask operations
            frame_result = super().apply_mask_operation(
                processed_mask[np.newaxis, ...],  # Add batch dimension
                original_masks[i:i+1],  # Single frame
                current_mask_strength,
                invert,
                current_subtract,
                current_blur
            )
            result.append(frame_result)
            # self.update_progress()

        # self.end_progress()

        # Stack all frames back together
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
