from .mask_base import FlexMaskBase
from .mask_utils import morph_mask, warp_mask, transform_mask, combine_masks
import numpy as np

class FlexMaskMorph(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "morph_type": (["erode", "dilate", "open", "close"],),
                "max_kernel_size": ("INT", {"default": 5, "min": 3, "max": 21, "step": 2}),
                "max_iterations": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
            }
        }

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, morph_type: str, max_kernel_size: int, max_iterations: int, **kwargs) -> np.ndarray:
        kernel_size = max(3, int(3 + (max_kernel_size - 3) * feature_value * strength))
        iterations = max(1, int(max_iterations * feature_value * strength))
        
        return morph_mask(mask, morph_type, kernel_size, iterations)

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, morph_type, max_kernel_size, max_iterations, **kwargs):
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, 
                                          morph_type=morph_type, max_kernel_size=max_kernel_size, max_iterations=max_iterations, **kwargs),)

class FlexMaskWarp(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "warp_type": (["perlin", "radial", "swirl"],),
                "frequency": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "max_amplitude": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 500.0, "step": 0.1}),
                "octaves": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
            }
        }

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, warp_type: str, frequency: float, max_amplitude: float, octaves: int, **kwargs) -> np.ndarray:
        amplitude = max_amplitude * feature_value * strength
        return warp_mask(mask, warp_type, frequency, amplitude, octaves)

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, warp_type, frequency, max_amplitude, octaves, **kwargs):
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, 
                                          warp_type=warp_type, frequency=frequency, max_amplitude=max_amplitude, octaves=octaves, **kwargs),)

class FlexMaskTransform(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "transform_type": (["translate", "rotate", "scale"],),
                "max_x_value": ("FLOAT", {"default": 10.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "max_y_value": ("FLOAT", {"default": 10.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
            }
        }

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, transform_type: str, max_x_value: float, max_y_value: float, **kwargs) -> np.ndarray:
        x_value = max_x_value * feature_value * strength
        y_value = max_y_value * feature_value * strength
        return transform_mask(mask, transform_type, x_value, y_value)

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, transform_type, max_x_value, max_y_value, **kwargs):
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, 
                                          transform_type=transform_type, max_x_value=max_x_value, max_y_value=max_y_value, **kwargs),)

class FlexMaskMath(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "mask_b": ("MASK",),
                "combination_method": (["add", "subtract", "multiply", "minimum", "maximum"],),
            }
        }

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, mask_b: np.ndarray, combination_method: str, **kwargs) -> np.ndarray:
        return combine_masks(mask, mask_b, combination_method, feature_value * strength)

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, mask_b, combination_method, **kwargs):
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, 
                                          mask_b=mask_b, combination_method=combination_method, **kwargs),)