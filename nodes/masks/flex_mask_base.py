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

    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        return ["None"]

    @abstractmethod
    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
                    frame_index: int = 0, **kwargs) -> np.ndarray:
        # Process parameters using base class functionality
        processed_kwargs = self.process_parameters(
            frame_index=frame_index,
            feature_value=feature_value,
            strength=strength,
            feature_param=kwargs.get('feature_param'),
            feature_mode=kwargs.get('feature_mode'),
            **kwargs
        )
        
        # Call the child class's implementation with processed parameters
        return self.apply_effect_internal(mask, **processed_kwargs)

    def apply_effect_internal(self, mask: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect with processed parameters."""
        return self.process_mask(mask, **kwargs)

    def apply_effect(self, masks, opt_feature=None, strength=1.0, feature_threshold=0.0, mask_strength=1.0, invert=False, subtract_original=0.0, grow_with_blur=0.0, **kwargs):
        """Main entry point for the Flex system.
        
        This method implements the required FlexBase.apply_effect method and routes to our mask-specific implementation.
        """
        num_frames = masks.shape[0]
        original_masks = masks.clone()

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        result = []
        for i in range(num_frames):
            kwargs['frame_index'] = i
            mask = masks[i].numpy()
            
            # When feature_param is "None", always apply the effect with default feature value
            if kwargs.get('feature_param') == "None":
                feature_value = 0.5  # Default feature value
                kwargs['feature_value'] = feature_value
                kwargs['strength'] = strength
                processed_mask = self.apply_effect_internal(mask, **kwargs)
            else:
                # Normal feature-based behavior
                if opt_feature is not None:
                    feature_value = opt_feature.get_value_at_frame(i)
                    apply_effect = feature_value >= feature_threshold
                else:
                    feature_value = 0.5  # Default feature value when no feature is provided
                    apply_effect = True

                if apply_effect:
                    kwargs['feature_value'] = feature_value
                    kwargs['strength'] = strength
                    processed_mask = self.apply_effect_internal(mask, **kwargs)
                else:
                    if hasattr(self, 'process_mask_below_threshold'):
                        kwargs['feature_value'] = feature_value
                        kwargs['strength'] = strength
                        processed_mask = self.process_mask_below_threshold(mask, **kwargs)
                    else:
                        processed_mask = mask

            result.append(processed_mask)
            self.update_progress()

        self.end_progress()

        processed_masks = torch.from_numpy(np.stack(result)).float()
        
        # Remove strength from kwargs to avoid duplicate argument
        kwargs_for_mask_op = kwargs.copy()
        kwargs_for_mask_op.pop('strength', None)
        kwargs_for_mask_op.pop('feature_value', None)
        
        # Use mask_strength instead of strength for the final mask operation
        return (super().apply_mask_operation(processed_masks, original_masks, mask_strength, invert, subtract_original, grow_with_blur, **kwargs_for_mask_op),)

    def main_function(self, masks, opt_feature=None, strength=1.0, feature_threshold=0.0, mask_strength=1.0, invert=False, subtract_original=0.0, grow_with_blur=0.0, **kwargs):
        """Implementation of MaskBase's abstract main_function.
        
        Explicitly forwards all parameters to apply_effect to maintain parameter names.
        """
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