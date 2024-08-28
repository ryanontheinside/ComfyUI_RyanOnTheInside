from .mask_base import FlexMaskBase
from .mask_utils import morph_mask, warp_mask, transform_mask, combine_masks
import math
import numpy as np
from .voronoi_noise import VoronoiNoise #NOTE credit for Voronoi goes to Alan Huang https://github.com/alanhuang67/
from comfy.model_management import get_torch_device

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
    
class FlexMaskOpacity(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "max_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, max_opacity: float, **kwargs) -> np.ndarray:
        opacity = max_opacity * feature_value * strength
        return mask * opacity

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, max_opacity, **kwargs):
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, 
                                          max_opacity=max_opacity, **kwargs),)
    
#NOTE credit for the heavy lifting in this class and all of  the noise class goes to Alan Huang https://github.com/alanhuang67/
class FlexMaskVoronoiScheduled(FlexMaskBase):
    formulas = {
        "Linear": lambda t, a, b: t * a / b,
        "Quadratic": lambda t, a, b: (t * a / b) ** 2,
        "Cubic": lambda t, a, b: (t * a / b) ** 3,
        "Sinusoidal": lambda t, a, b: math.sin(math.pi * t * a / b / 2),
        "Exponential": lambda t, a, b: math.exp(t * a / b) - 1,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "distance_metric": ([
                    "euclidean", "manhattan", "chebyshev", "minkowski",
                    "elliptical", "kaleidoscope_star", "kaleidoscope_wave",
                    "kaleidoscope_radiation_α", "kaleidoscope_radiation_β",
                    "kaleidoscope_radiation_γ"
                ],),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "detail": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
                "randomness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "x_offset": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "y_offset": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "control_parameter": (["scale", "detail", "randomness", "seed", "x_offset", "y_offset"],),
                "formula": (list(cls.formulas.keys()),),
                "a": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "b": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    def generate_schedule(self, formula, feature_value, a, b):
        t = feature_value
        return self.formulas[formula](t, a, b)

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
                     distance_metric: str, scale: float, detail: int, randomness: float, 
                     seed: int, x_offset: float, y_offset: float, control_parameter: str,
                     formula: str, a: float, b: float, **kwargs) -> np.ndarray:
        
        height, width = mask.shape[:2]

        # Generate schedule value
        schedule_value = self.generate_schedule(formula, feature_value, a, b)

        # Adjust the controlled parameter based on the schedule value and strength
        if control_parameter == "scale":
            scale *= (1 + schedule_value * strength)
        elif control_parameter == "detail":
            detail = int(detail * (1 + schedule_value * strength))
        elif control_parameter == "randomness":
            randomness *= (1 + schedule_value * strength)
        elif control_parameter == "seed":
            seed = int(seed + (schedule_value * strength * 1000000))
        elif control_parameter == "x_offset":
            x_offset *= (1 + schedule_value * strength)
        elif control_parameter == "y_offset":
            y_offset *= (1 + schedule_value * strength)

        # Create VoronoiNoise instance
        voronoi = VoronoiNoise(
            width=width, 
            height=height, 
            scale=[scale], 
            detail=[detail], 
            seed=[seed], 
            randomness=[randomness],
            X=[x_offset],
            Y=[y_offset],
            distance_metric=distance_metric,
            batch_size=1,
            device=get_torch_device()
        )

        # Generate Voronoi noise
        voronoi_tensor = voronoi()

        # Convert to numpy array and extract the first channel (they're all the same)
        voronoi_mask = voronoi_tensor[0, :, :, 0].cpu().numpy()

        return voronoi_mask

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, 
                      invert, subtract_original, grow_with_blur, distance_metric, 
                      scale, detail, randomness, seed, x_offset, y_offset, 
                      control_parameter, formula, a, b, **kwargs):
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength, 
                                          feature_threshold, invert, subtract_original, 
                                          grow_with_blur, distance_metric=distance_metric, 
                                          scale=scale, detail=detail, randomness=randomness, 
                                          seed=seed, x_offset=x_offset, y_offset=y_offset, 
                                          control_parameter=control_parameter, 
                                          formula=formula, a=a, b=b, **kwargs),)

#TODO
# class FlexMaskVoronoiShape(FlexMaskBase):
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             **super().INPUT_TYPES(),
#             "required": {
#                 **super().INPUT_TYPES()["required"],
#                 "max_num_points": ("INT", {"default": 50, "min": 4, "max": 1000, "step": 1}),
#                 "max_point_jitter": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
#                 "shape_type": (get_available_shapes(),),
#                 "max_shape_size": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
#                 "control_mode": (["num_points", "point_jitter", "shape_size", "all"],),
#             }
#         }

#     def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
#                      max_num_points: int, max_point_jitter: float, shape_type: str, 
#                      max_shape_size: float, control_mode: str, **kwargs) -> np.ndarray:
#         if control_mode == "num_points" or control_mode == "all":
#             num_points = max(4, int(max_num_points * feature_value * strength))
#         else:
#             num_points = max_num_points

#         if control_mode == "point_jitter" or control_mode == "all":
#             point_jitter = max_point_jitter * feature_value * strength
#         else:
#             point_jitter = max_point_jitter

#         if control_mode == "shape_size" or control_mode == "all":
#             shape_size = max_shape_size * feature_value * strength
#         else:
#             shape_size = max_shape_size
        
#         return generate_voronoi_shapes_mask(mask.shape[:2], num_points, point_jitter, shape_type, shape_size)

#     def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, 
#                       invert, subtract_original, grow_with_blur, max_num_points, 
#                       max_point_jitter, shape_type, max_shape_size, control_mode, **kwargs):
#         return (self.apply_mask_operation(masks, feature, feature_pipe, strength, 
#                                           feature_threshold, invert, subtract_original, 
#                                           grow_with_blur, max_num_points=max_num_points, 
#                                           max_point_jitter=max_point_jitter, 
#                                           shape_type=shape_type, 
#                                           max_shape_size=max_shape_size, 
#                                           control_mode=control_mode, **kwargs),)