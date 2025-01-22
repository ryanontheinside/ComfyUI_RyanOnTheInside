from .mask_base import TemporalMaskBase
from .mask_utils import morph_mask, transform_mask, combine_masks, warp_mask
import numpy as np
import torch
import cv2
from scipy.ndimage import distance_transform_edt
from ...tooltips import apply_tooltips


@apply_tooltips
class MaskMorph(TemporalMaskBase):
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

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply_mask_morph"


    def process_single_mask(self, mask: np.ndarray, strength: float, morph_type: str, max_kernel_size: int, max_iterations: int, **kwargs) -> np.ndarray:
        # Scale kernel size and iterations based on strength
        kernel_size = max(3, int(3 + (max_kernel_size - 3) * strength))
        iterations = max(1, int(max_iterations * strength))
        
        return morph_mask(mask, morph_type, kernel_size, iterations)

    def apply_mask_morph(self, masks, strength, morph_type, max_kernel_size, max_iterations, **kwargs):
        return super().main_function(masks, strength, morph_type=morph_type, max_kernel_size=max_kernel_size, max_iterations=max_iterations, **kwargs)

@apply_tooltips
class MaskTransform(TemporalMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "transform_type": (["translate", "rotate", "scale"],),
                "x_value": ("FLOAT", {"default": 0, "min": -1000, "max": 1000, "step": 0.1}),
                "y_value": ("FLOAT", {"default": 0, "min": -1000, "max": 1000, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply_mask_transform"


    def process_single_mask(self, mask: np.ndarray, strength: float, transform_type: str, x_value: float, y_value: float, **kwargs) -> np.ndarray:
        transformed_mask = transform_mask(mask, transform_type, x_value * strength, y_value * strength)
        return transformed_mask

    def apply_mask_transform(self, masks, strength, transform_type, x_value, y_value, **kwargs):
        return super().main_function(masks, strength, transform_type=transform_type, x_value=x_value, y_value=y_value, **kwargs)
    
@apply_tooltips
class MaskMath(TemporalMaskBase):
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

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply_mask_math"

    
    def process_single_mask(self, mask: np.ndarray, strength: float, mask_b: np.ndarray, combination_method: str, frame_index: int, **kwargs) -> np.ndarray:
        mask_b_frame = mask_b[frame_index]
        return combine_masks(mask, mask_b_frame, combination_method, strength)

    def apply_mask_math(self, masks, mask_b, strength, combination_method, **kwargs):
        mask_b_np = mask_b.cpu().numpy() if isinstance(mask_b, torch.Tensor) else mask_b
        return super().main_function(masks, strength, mask_b=mask_b_np, combination_method=combination_method, **kwargs)
    
    #TODO CONFIRM THAT NOTHING HAPPENS WITH EMPTY MASK
@apply_tooltips
class MaskRings(TemporalMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "num_rings": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "max_ring_width": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply_mask_rings"


    def process_single_mask(self, mask: np.ndarray, strength: float, num_rings: int, max_ring_width: float, **kwargs) -> np.ndarray:
        distance = distance_transform_edt(1 - mask)
        max_distance = np.max(distance)

        normalized_distance = distance / max_distance

        # Create rings
        rings = np.zeros_like(mask)
        for i in range(num_rings):
            ring_progress = max(0, min(1, num_rings * strength - i))
            if ring_progress > 0:
                ring_width = max_ring_width * ring_progress
                ring_outer = normalized_distance < (i + 1) / num_rings
                ring_inner = normalized_distance < (i + 1 - ring_width) / num_rings
                rings = np.logical_or(rings, np.logical_xor(ring_outer, ring_inner))

        # Combine with original mask
        result = np.logical_or(mask, rings).astype(np.float32)

        return result

    def apply_mask_rings(self, masks, strength, num_rings, max_ring_width, **kwargs):
        return super().main_function(masks, strength, num_rings=num_rings, max_ring_width=max_ring_width, **kwargs)

@apply_tooltips
class MaskWarp(TemporalMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "warp_type": (["perlin", "radial", "swirl"],),
                "frequency": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "amplitude": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 500.0, "step": 0.1}),
                "octaves": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply_mask_warp"

    def process_single_mask(self, mask: np.ndarray, strength: float, warp_type: str, frequency: float, amplitude: float, octaves: int, **kwargs) -> np.ndarray:
        return warp_mask(mask, warp_type, frequency, amplitude * strength, octaves)

    def apply_mask_warp(self, masks, strength, warp_type, frequency, amplitude, octaves, **kwargs):
        return super().main_function(masks, strength, warp_type=warp_type, frequency=frequency, amplitude=amplitude, octaves=octaves, **kwargs)