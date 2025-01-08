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
        # Get input types from both parent classes
        flex_inputs = FlexBase.INPUT_TYPES()
        mask_inputs = MaskBase.INPUT_TYPES()
        
        # First rename MaskBase's strength to mask_strength
        mask_inputs["required"]["mask_strength"] = mask_inputs["required"].pop("strength")
        
        # Then merge the inputs
        required = {
            **flex_inputs["required"],
            **mask_inputs["required"]
        }
        
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
        # Initialize scheduler if needed
        if self.parameter_scheduler is None:
            frame_count = kwargs.get('frame_count', 1)
            for value in kwargs.values():
                if isinstance(value, (list, tuple, np.ndarray)):
                    frame_count = max(frame_count, len(value))
            self.initialize_scheduler(frame_count, **kwargs)

        # Process each parameter for this frame
        processed = {}
        for key, value in kwargs.items():
            # Handle array-like parameters
            if isinstance(value, (list, tuple, np.ndarray)):
                try:
                    processed[key] = float(value[frame_index])
                except (IndexError, TypeError):
                    processed[key] = float(value[0])
            else:
                processed[key] = value

        # Handle feature modulation for the selected parameter
        feature_param = kwargs.get('feature_param')
        feature_mode = kwargs.get('feature_mode', 'relative')
        
        if feature_param and feature_param in processed and feature_param != "None":
            # processed[feature_param] is already a single value at this point
            processed[feature_param] = self.modulate_param(
                feature_param,
                processed[feature_param],  # Already converted to float above
                feature_value,
                strength,
                feature_mode
            )

        return processed

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
                    frame_index: int = 0, **kwargs) -> np.ndarray:
        """Process a single mask with feature modulation and parameter scheduling."""
        # Get the processed parameters for this frame
        processed_kwargs = self.process_parameters(
            frame_index=frame_index,
            feature_value=feature_value,
            strength=strength,
            feature_param=kwargs.get('feature_param'),
            feature_mode=kwargs.get('feature_mode'),
            **kwargs
        )
        
        # Remove parameters that shouldn't be passed to child classes
        feature_param = processed_kwargs.pop('feature_param', None)
        feature_mode = processed_kwargs.pop('feature_mode', None)
        processed_kwargs.pop('frame_index', None)
        
        # If this parameter is being modulated by a feature, handle it separately
        if feature_param in processed_kwargs:
            base_value = processed_kwargs[feature_param]
            if isinstance(base_value, (list, tuple, np.ndarray)):
                try:
                    base_value = float(base_value[frame_index])
                except (IndexError, TypeError):
                    base_value = float(base_value[0])
            processed_kwargs[feature_param] = self.modulate_param(
                feature_param, 
                float(base_value), 
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

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        # Initialize parameter scheduler if needed
        if self.parameter_scheduler is None:
            frame_count = num_frames
            self.initialize_scheduler(frame_count, 
                                   mask_strength=mask_strength,
                                   subtract_original=subtract_original,
                                   grow_with_blur=grow_with_blur,
                                   **kwargs)

        result = []
        for i in range(num_frames):
            mask = masks[i].numpy()
            
            # Get feature value and determine if effect should be applied
            feature_value = self.get_feature_value(opt_feature, i)
            apply_effect = feature_value >= feature_threshold if opt_feature is not None else True

            if apply_effect:
                processed_mask = self.process_mask(
                    mask,
                    feature_value=feature_value,
                    strength=strength,
                    frame_index=i,
                    **kwargs
                )
            else:
                if hasattr(self, 'process_mask_below_threshold'):
                    processed_mask = self.process_mask_below_threshold(
                        mask,
                        feature_value=feature_value,
                        strength=strength,
                        frame_index=i,
                        **kwargs
                    )
                else:
                    processed_mask = mask

            # Get scheduled values for mask parameters for this frame
            if self.parameter_scheduler:
                current_mask_strength = float(self.parameter_scheduler.get_value('mask_strength', i) if self.parameter_scheduler.is_scheduled('mask_strength') else mask_strength)
                current_subtract = float(self.parameter_scheduler.get_value('subtract_original', i) if self.parameter_scheduler.is_scheduled('subtract_original') else subtract_original)
                current_blur = float(self.parameter_scheduler.get_value('grow_with_blur', i) if self.parameter_scheduler.is_scheduled('grow_with_blur') else grow_with_blur)
            else:
                # Handle case where inputs might be lists but scheduler isn't initialized
                current_mask_strength = float(mask_strength[i] if isinstance(mask_strength, (list, tuple, np.ndarray)) else mask_strength)
                current_subtract = float(subtract_original[i] if isinstance(subtract_original, (list, tuple, np.ndarray)) else subtract_original)
                current_blur = float(grow_with_blur[i] if isinstance(grow_with_blur, (list, tuple, np.ndarray)) else grow_with_blur)

            # Apply mask operations for this frame
            frame_result = super().apply_mask_operation(
                processed_mask[np.newaxis, ...],  # Add batch dimension
                original_masks[i:i+1],  # Single frame
                current_mask_strength,
                invert,
                current_subtract,
                current_blur
            )
            result.append(frame_result)

        self.end_progress()

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