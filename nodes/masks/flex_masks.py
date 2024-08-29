from .mask_base import FlexMaskBase
from .mask_utils import morph_mask, warp_mask, transform_mask, combine_masks
import math
import numpy as np
from .voronoi_noise import VoronoiNoise #NOTE credit for Voronoi goes to Alan Huang https://github.com/alanhuang67/
from comfy.model_management import get_torch_device
import cv2

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
                "feature_param": (["scale", "detail", "randomness", "seed", "x_offset", "y_offset"],),
                "formula": (list(cls.formulas.keys()),),
                "x_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "y_offset": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    def generate_schedule(self, formula, feature_value, a, b):
        t = feature_value
        return self.formulas[formula](t, a, b)

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
                     distance_metric: str, scale: float, detail: int, randomness: float, 
                     seed: int, x_offset: float, y_offset: float, feature_param: str,
                     formula: str, a: float, b: float, **kwargs) -> np.ndarray:
        
        height, width = mask.shape[:2]

        # Generate schedule value
        schedule_value = self.generate_schedule(formula, feature_value, a, b)

        # Adjust the controlled parameter based on the schedule value and strength
        if feature_param == "scale":
            scale *= (1 + schedule_value * strength)
        elif feature_param == "detail":
            detail = int(detail * (1 + schedule_value * strength))
        elif feature_param == "randomness":
            randomness *= (1 + schedule_value * strength)
        elif feature_param == "seed":
            seed = int(seed + (schedule_value * strength * 1000000))
        elif feature_param == "x_offset":
            x_offset += width * schedule_value * strength
        elif feature_param == "y_offset":
            y_offset += height * schedule_value * strength

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
                      feature_param, formula, a, b, **kwargs):
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength, 
                                          feature_threshold, invert, subtract_original, 
                                          grow_with_blur, distance_metric=distance_metric, 
                                          scale=scale, detail=detail, randomness=randomness, 
                                          seed=seed, x_offset=x_offset, y_offset=y_offset, 
                                          feature_param=feature_param, 
                                          formula=formula, a=a, b=b, **kwargs),)

class FlexMaskBinary(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "method": (["simple", "adaptive", "hysteresis", "edge"],),
                "max_smoothing": ("INT", {"default": 21, "min": 0, "max": 51, "step": 2}),
                "max_edge_enhancement": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "feature_param": (["threshold", "none", "smoothing", "edge_enhancement"],),
                "use_epsilon": ("BOOLEAN", {"default": False}),
            }
        }

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, threshold: float, 
                     method: str, max_smoothing: int, max_edge_enhancement: float, 
                     feature_param: str, use_epsilon: bool, **kwargs) -> np.ndarray:
        mask = mask.astype(np.float32)
        mask = np.clip(mask, 0, 1)

        # Apply smoothing
        if feature_param == "smoothing":
            smoothing = int(max_smoothing * feature_value * strength)
        else:
            smoothing = int(max_smoothing * 0.5)
        
        if smoothing > 0:
            mask = cv2.GaussianBlur(mask, (smoothing * 2 + 1, smoothing * 2 + 1), 0)

        # Apply edge enhancement
        if feature_param == "edge_enhancement":
            edge_enhancement = max_edge_enhancement * feature_value * strength
        else:
            edge_enhancement = max_edge_enhancement * 0.5
        
        if edge_enhancement > 0:
            laplacian = cv2.Laplacian(mask, cv2.CV_32F, ksize=3)
            mask = np.clip(mask + edge_enhancement * laplacian, 0, 1)

        # Adjust threshold
        if feature_param == "threshold":
            adjusted_threshold = threshold + (feature_value - 0.5) * strength * 0.5
        else:
            adjusted_threshold = threshold
        adjusted_threshold = max(0.0, min(1.0, adjusted_threshold))

        if method == "simple":
            if use_epsilon:
                epsilon = 1e-7  # Small value to avoid exact comparisons
                binary_mask = ((mask > adjusted_threshold + epsilon) | 
                               (abs(mask - adjusted_threshold) < epsilon)).astype(np.float32)
            else:
                binary_mask = (mask > adjusted_threshold).astype(np.float32)
        elif method == "adaptive":
            mask_uint8 = (mask * 255).astype(np.uint8)
            binary_mask = cv2.adaptiveThreshold(
                mask_uint8,
                1,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,  # block size
                2    # C constant
            ).astype(np.float32)
        elif method == "hysteresis":
            low_threshold = max(0, adjusted_threshold - 0.1)
            high_threshold = min(1, adjusted_threshold + 0.1)
            low_mask = mask > low_threshold
            high_mask = mask > high_threshold
            binary_mask = cv2.connectedComponents((high_mask * 255).astype(np.uint8))[1]
            binary_mask = ((binary_mask > 0) & low_mask).astype(np.float32)
        elif method == "edge":
            mask_uint8 = (mask * 255).astype(np.uint8)
            edges = cv2.Canny(mask_uint8, 
                              int(adjusted_threshold * 255 * 0.5), 
                              int(adjusted_threshold * 255 * 1.5))
            binary_mask = edges.astype(np.float32) / 255.0

        return binary_mask

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, invert, 
                      subtract_original, grow_with_blur, threshold, method, max_smoothing, 
                      max_edge_enhancement, feature_param, use_epsilon, **kwargs):
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength, feature_threshold, 
                                          invert, subtract_original, grow_with_blur, 
                                          threshold=threshold, method=method, 
                                          max_smoothing=max_smoothing, 
                                          max_edge_enhancement=max_edge_enhancement, 
                                          feature_param=feature_param,
                                          use_epsilon=use_epsilon, **kwargs),)



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