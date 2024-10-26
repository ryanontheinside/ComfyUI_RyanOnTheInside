from .flex_mask_base import FlexMaskBase
from .mask_utils import morph_mask, warp_mask, transform_mask, combine_masks,apply_easing
import math
import numpy as np
from .voronoi_noise import VoronoiNoise #NOTE credit for Voronoi goes to Alan Huang https://github.com/alanhuang67/
from comfy.model_management import get_torch_device
import cv2
from scipy.ndimage import distance_transform_edt
from .shape_utils import create_shape_mask, get_available_shapes
import torch
from typing import List
import torch.nn.functional as F

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
        frame_index = kwargs.get('frame_index', 0)
        mask_b = mask_b[frame_index].numpy()
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
                "a": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "b": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
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

class FlexMaskWavePropagation(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        cls.feature_threshold_default = 0.25
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "wave_speed": ("FLOAT", {"default": 50.0, "min": 0.1, "max": 100.0, "step": 0.5}),
                "wave_amplitude": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
                "wave_decay": ("FLOAT", {"default": 5.0, "min": 0.9, "max": 10.0, "step": 0.001}),
                "wave_frequency": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 10.0, "step": 0.01}),
                "max_wave_field": ("FLOAT", {"default": 750.0, "min": 10.0, "max": 10000.0, "step": 10.0}),
            }
        }

    def __init__(self):
        super().__init__()
        self.wave_field = None
        self.frame_count = 0

    def process_mask_below_threshold(self, mask, feature_value, strength, **kwargs):
        self.wave_field = None
        self.frame_count = 0
        return mask

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
                     wave_speed: float, wave_amplitude: float, wave_decay: float, 
                     wave_frequency: float, max_wave_field: float, **kwargs) -> np.ndarray:
        height, width = mask.shape
        
        if self.wave_field is None:
            self.wave_field = np.zeros((height, width), dtype=np.float32)
        
        # Find mask boundary
        kernel = np.ones((3,3), np.uint8)
        boundary = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) - mask.astype(np.uint8)
        
        # Reset wave field where the mask is not present
        self.wave_field[mask == 0] *= wave_decay
        
        # Emit wave from boundary and propagate
        self.wave_field += boundary * feature_value * wave_amplitude
        self.wave_field = cv2.GaussianBlur(self.wave_field, (0, 0), sigmaX=wave_speed)
        
        # Apply decay
        self.wave_field *= wave_decay
        
        # Normalize wave field if it exceeds max_wave_field
        max_value = np.max(np.abs(self.wave_field))
        if max_value > max_wave_field:
            self.wave_field *= (max_wave_field / max_value)
        
        time_factor = self.frame_count * wave_frequency
        wave_pattern = np.sin(self.wave_field + time_factor) * 0.5 + 0.5
        
        # Combine with original mask
        result_mask = np.clip(mask + wave_pattern * strength, 0, 1)
        
        # Print debug information
        print(f"Frame: {self.frame_count}")
        print(f"Wave field min/max: {self.wave_field.min():.4f} / {self.wave_field.max():.4f}")
        print(f"Wave pattern min/max: {wave_pattern.min():.4f} / {wave_pattern.max():.4f}")
        print(f"Result mask min/max: {result_mask.min():.4f} / {result_mask.max():.4f}")
        print("---")
        
        self.frame_count += 1
        
        return result_mask.astype(np.float32)

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, 
                      invert, subtract_original, grow_with_blur, **kwargs):
        # Reset wave_field and frame_count for each new feature input
        self.wave_field = None
        self.frame_count = 0
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength, 
                                          feature_threshold, invert, subtract_original, 
                                          grow_with_blur, **kwargs),)

class FlexMaskEmanatingRings(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        cls.feature_threshold_default = 0.25
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "num_rings": ("INT", {"default": 4, "min": 1, "max": 50, "step": 1}),
                "max_ring_width": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.9, "step": 0.01}),
                "wave_speed": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01}),
                "feature_param": (["num_rings", "ring_width", "wave_speed", "all"],),
            }
        }

    def __init__(self):
        super().__init__()
        self.rings = []

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float,
                     num_rings: int, max_ring_width: float, wave_speed: float,
                     feature_param: str, **kwargs) -> np.ndarray:
        height, width = mask.shape
        distance = distance_transform_edt(1 - mask)
        max_distance = np.max(distance)
        normalized_distance = distance / max_distance

        # Update existing rings
        new_rings = []
        for ring in self.rings:
            ring['progress'] += ring['wave_speed']
            if ring['progress'] < 1:
                new_rings.append(ring)
        self.rings = new_rings

        # Create new rings if feature_value > 0
        if feature_value > 0:
            if feature_param in ["num_rings", "all"]:
                adjusted_num_rings = max(1, int(num_rings * feature_value * strength))
            else:
                adjusted_num_rings = num_rings

            if feature_param in ["ring_width", "all"]:
                adjusted_max_ring_width = max_ring_width * feature_value * strength
            else:
                adjusted_max_ring_width = max_ring_width

            if feature_param in ["wave_speed", "all"]:
                adjusted_wave_speed = wave_speed * feature_value * strength
            else:
                adjusted_wave_speed = wave_speed

            for i in range(adjusted_num_rings):
                self.rings.append({
                    'progress': i / adjusted_num_rings,
                    'ring_width': adjusted_max_ring_width,
                    'wave_speed': adjusted_wave_speed
                })

        # Create emanating rings
        rings = np.zeros_like(mask)
        for ring in self.rings:
            ring_progress = ring['progress'] % 1
            ring_width = ring['ring_width'] * (1 - ring_progress)  # Rings get thinner as they move out
            ring_outer = normalized_distance < ring_progress
            ring_inner = normalized_distance < (ring_progress - ring_width)
            rings = np.logical_or(rings, np.logical_xor(ring_outer, ring_inner))

        # Combine with original mask
        result = np.logical_or(mask, rings).astype(np.float32)

        return result

    def process_mask_below_threshold(self, mask: np.ndarray, feature_value: float, strength: float, **kwargs) -> np.ndarray:
        # Continue the animation but don't create new rings
        return self.process_mask(mask, 0, strength, **kwargs)

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold,
                      invert, subtract_original, grow_with_blur, num_rings,
                      max_ring_width, wave_speed, feature_param, **kwargs):
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength,
                                          feature_threshold, invert, subtract_original,
                                          grow_with_blur, num_rings=num_rings,
                                          max_ring_width=max_ring_width, wave_speed=wave_speed,
                                          feature_param=feature_param, **kwargs),)

class FlexMaskRandomShapes(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        cls.feature_threshold_default = 0.25
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "max_num_shapes": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "max_shape_size": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 1.0, "step": 0.01}),
                "appearance_duration": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "disappearance_duration": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
                "appearance_method": (["grow", "pop", "fade"],),
                "easing_function": (["linear","ease_in_out", "bounce","elastic"],),
                "shape_type": (get_available_shapes(),),
                "feature_param": (["num_shapes", "shape_size", "appearance_duration", "disappearance_duration"],),
            }
        }

    def __init__(self):
        super().__init__()
        self.shapes = []
        self.frame_count = 0

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float,
                     max_num_shapes: int, max_shape_size: float, appearance_duration: int,
                     disappearance_duration: int, appearance_method: str, easing_function: str,
                     shape_type: str, feature_param: str, **kwargs) -> np.ndarray:
        height, width = mask.shape
        result_mask = mask.copy()

        # Adjust parameters based on feature_value and feature_param
        if feature_param == "num_shapes":
            num_shapes = max(1, int(max_num_shapes * feature_value * strength))
        else:
            num_shapes = max_num_shapes

        if feature_param == "shape_size":
            shape_size = max_shape_size * feature_value * strength
        else:
            shape_size = max_shape_size

        if feature_param == "appearance_duration":
            app_duration = max(1, int(appearance_duration * feature_value * strength))
        else:
            app_duration = appearance_duration

        if feature_param == "disappearance_duration":
            disapp_duration = max(1, int(disappearance_duration * feature_value * strength))
        else:
            disapp_duration = disappearance_duration

        # Remove completed shapes
        self.shapes = [shape for shape in self.shapes if shape['frame'] < shape['total_frames']]

        # Add new shapes if needed
        while len(self.shapes) < num_shapes:
            center = (np.random.randint(0, width), np.random.randint(0, height))
            if shape_type == "random":
                selected_shape = np.random.choice(get_available_shapes())
            else:
                selected_shape = shape_type
            new_shape = {
                'center': center,
                'size': int(min(height, width) * shape_size),
                'type': selected_shape,
                'frame': 0,
                'total_frames': app_duration + disapp_duration,
                'app_duration': app_duration,
                'disapp_duration': disapp_duration,
            }
            self.shapes.append(new_shape)

        # Update and draw shapes
        for shape in self.shapes:
            if shape['frame'] < shape['app_duration']:
                progress = shape['frame'] / shape['app_duration']
                alpha = apply_easing(progress, easing_function)
                if appearance_method == "grow":
                    current_size = int(shape['size'] * alpha)
                    shape_mask = create_shape_mask((height, width), shape['center'], shape['type'], current_size)
                elif appearance_method == "pop":
                    shape_mask = create_shape_mask((height, width), shape['center'], shape['type'], shape['size']) * (1 if progress > 0.5 else 0)
                elif appearance_method == "fade":
                    shape_mask = create_shape_mask((height, width), shape['center'], shape['type'], shape['size']) * alpha
            else:
                progress = (shape['frame'] - shape['app_duration']) / shape['disapp_duration']
                alpha = 1 - apply_easing(progress, easing_function)
                if appearance_method == "grow":
                    current_size = int(shape['size'] * (1 - progress))
                    shape_mask = create_shape_mask((height, width), shape['center'], shape['type'], current_size)
                elif appearance_method == "pop":
                    shape_mask = create_shape_mask((height, width), shape['center'], shape['type'], shape['size']) * (1 if progress < 0.5 else 0)
                elif appearance_method == "fade":
                    shape_mask = create_shape_mask((height, width), shape['center'], shape['type'], shape['size']) * alpha

            result_mask = np.maximum(result_mask, shape_mask)
            shape['frame'] += 1

        self.frame_count += 1
        return result_mask

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold,
                      invert, subtract_original, grow_with_blur, max_num_shapes,
                      max_shape_size, appearance_duration, disappearance_duration,
                      appearance_method, easing_function, shape_type, feature_param, **kwargs):
        
        self.shapes=[]
        self.frame_count=0
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength,
                                          feature_threshold, invert, subtract_original,
                                          grow_with_blur, max_num_shapes=max_num_shapes,
                                          max_shape_size=max_shape_size,
                                          appearance_duration=appearance_duration,
                                          disappearance_duration=disappearance_duration,
                                          appearance_method=appearance_method,
                                          easing_function=easing_function,
                                          shape_type=shape_type,
                                          feature_param=feature_param, **kwargs),)

class FlexMaskDepthChamber(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "depth_map": ("IMAGE",),
                "z_front": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "z_back": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_param": (["none", "z_front", "z_back", "both"],),
                "feature_mode": (["squeeze", "expand", "move_forward", "move_back"],),
            }
        }

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
                     depth_map: torch.Tensor, z_front: float, z_back: float, feature_param: str, 
                     feature_mode: str, **kwargs) -> np.ndarray:
        frame_index = kwargs.get('frame_index', 0)
        depth_map_frame = depth_map[frame_index].cpu().numpy()

        depth_map_frame = depth_map_frame[:, :, 0]

        # Adjust z_front and z_back based on feature_mode and feature_param
        if feature_param != "none":
            if feature_mode == "squeeze":
                if feature_param in ["z_front", "both"]:
                    z_front = z_front - (z_front - z_back) * strength * feature_value / 2 if z_front > z_back else z_front + (z_back - z_front) * strength * feature_value / 2
                if feature_param in ["z_back", "both"]:
                    z_back = z_back + (z_front - z_back) * strength * feature_value / 2 if z_back < z_front else z_back - (z_back - z_front) * strength * feature_value / 2
            elif feature_mode == "expand":
                if feature_param in ["z_front", "both"]:
                    z_front = min(1.0, z_front + (z_front - z_back) * strength * feature_value / 2) if z_front > z_back else max(0.0, z_front - (z_back - z_front) * strength * feature_value / 2)
                if feature_param in ["z_back", "both"]:
                    z_back = max(0.0, z_back - (z_front - z_back) * strength * feature_value / 2) if z_back < z_front else min(1.0, z_back + (z_back - z_front) * strength * feature_value / 2)
            elif feature_mode == "move_forward":
                if feature_param in ["z_front", "both"]:
                    z_front = min(1.0, z_front + strength * feature_value)
                if feature_param in ["z_back", "both"]:
                    z_back = min(1.0, z_back + strength * feature_value)
            elif feature_mode == "move_back":
                if feature_param in ["z_front", "both"]:
                    z_front = max(0.0, z_front - strength * feature_value)
                if feature_param in ["z_back", "both"]:
                    z_back = max(0.0, z_back - strength * feature_value)

        # Create the depth mask
        if z_back < z_front:
            depth_mask = (depth_map_frame >= z_back) & (depth_map_frame <= z_front)
        else:
            depth_mask = (depth_map_frame >= z_back) | (depth_map_frame <= z_front)

        depth_mask_resized = cv2.resize(depth_mask.astype(np.float32), (mask.shape[1], mask.shape[0]))

        # Subtract anything that doesn't fall within the input mask
        combined_mask = np.where(mask > 0, depth_mask_resized, 0)

        return combined_mask

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, 
                      invert, subtract_original, grow_with_blur, depth_map, z_front, z_back, 
                      feature_param, feature_mode, **kwargs):
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength, 
                                          feature_threshold, invert, subtract_original, 
                                          grow_with_blur, depth_map=depth_map, z_front=z_front, z_back=z_back, 
                                          feature_param=feature_param, feature_mode=feature_mode, 
                                          **kwargs),)

class FlexMaskDepthChamberRelative(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "depth_map": ("IMAGE",),
                "z1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "z2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_param": (["none", "z1", "z2", "both"],),
                "feature_mode": (["squeeze", "expand"],),
            }
        }

    def calculate_roi_size(self, mask: torch.Tensor) -> float:
        # Calculate the bounding box of the mask
        y_indices, x_indices = torch.where(mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return 0.0
        height = y_indices.max().item() - y_indices.min().item()
        width = x_indices.max().item() - x_indices.min().item()
        return height * width

    def calculate_reference_size(self, masks: List[torch.Tensor]) -> float:
        # Calculate the mean or median size of the ROI across all frames
        sizes = [self.calculate_roi_size(mask) for mask in masks]
        return torch.median(torch.tensor(sizes)).item()  # or torch.mean(torch.tensor(sizes)).item()

    def process_mask(self, mask: torch.Tensor, feature_value: float, strength: float, 
                     depth_map: torch.Tensor, z1: float, z2: float, feature_param: str, 
                     feature_mode: str, reference_size: float, **kwargs) -> torch.Tensor:
        frame_index = kwargs.get('frame_index', 0)
        depth_map_frame = depth_map[frame_index, :, :, 0]

        # Calculate the ROI size for the current frame
        roi_size = self.calculate_roi_size(mask)

        if feature_param == "z1":
            z1 = z1 * (roi_size / reference_size)
        elif feature_param == "z2":
            z2 = z2 * (roi_size / reference_size)
        elif feature_param == "both":
            z1 = z1 * (roi_size / reference_size)
            z2 = z2 * (roi_size / reference_size)

        # Ensure z1 is less than z2
        z1, z2 = min(z1, z2), max(z1, z2)

        if feature_mode == "squeeze":
            depth_mask = (depth_map_frame >= z1) & (depth_map_frame <= z2)
        elif feature_mode == "expand":
            depth_mask = (depth_map_frame < z1) | (depth_map_frame > z2)

        depth_mask_resized = F.interpolate(depth_mask.unsqueeze(0).unsqueeze(0).float(), size=mask.shape[-2:], mode='nearest').squeeze(0).squeeze(0)

        output_mask = mask.float() * depth_mask_resized

        return output_mask

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, 
                      invert, subtract_original, grow_with_blur, depth_map, z1, z2, 
                      feature_param, feature_mode, **kwargs):
        reference_size = self.calculate_reference_size(masks)

        output_masks = []
        for frame_index, mask in enumerate(masks):
            output_mask = self.process_mask(mask, feature_value=1.0, strength=1.0, 
                                            depth_map=depth_map, z1=z1, z2=z2, 
                                            feature_param=feature_param, feature_mode=feature_mode, 
                                            reference_size=reference_size, frame_index=frame_index)
            output_masks.append(output_mask)

        # Stack the list of tensors into a single tensor
        output_masks_tensor = torch.stack(output_masks)

        return (output_masks_tensor,)

class FlexMaskInterpolate(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "mask_b": ("MASK",),
                "interpolation_method": ([
                    "linear", "ease_in", "ease_out", "ease_in_out",
                    "cubic", "sigmoid", "radial",
                    "distance_transform", "random_noise"
                ],),
                "invert_mask_b": ("BOOLEAN", {"default": False}),
                "blend_mode": (["normal", "add", "multiply", "overlay", "soft_light"],),
            }
        }

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
                     mask_b: torch.Tensor, interpolation_method: str, invert_mask_b: bool,
                     blend_mode: str, **kwargs) -> np.ndarray:
        frame_index = kwargs.get('frame_index', 0)
        mask_b_frame = mask_b[frame_index].numpy()

        if invert_mask_b:
            mask_b_frame = 1.0 - mask_b_frame

        # Ensure masks are in the same shape
        if mask.shape != mask_b_frame.shape:
            mask_b_frame = cv2.resize(mask_b_frame, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Compute interpolation alpha based on feature_value and strength
        alpha = np.clip(feature_value * strength, 0.0, 1.0)

        # Apply interpolation method to compute weight
        if interpolation_method == "linear":
            weight = alpha
        elif interpolation_method == "ease_in":
            weight = alpha ** 2
        elif interpolation_method == "ease_out":
            weight = 1 - (1 - alpha) ** 2
        elif interpolation_method == "ease_in_out":
            weight = alpha ** 2 / (alpha ** 2 + (1 - alpha) ** 2 + 1e-6)
        elif interpolation_method == "cubic":
            weight = 3 * alpha ** 2 - 2 * alpha ** 3
        elif interpolation_method == "sigmoid":
            weight = 1 / (1 + np.exp(-12 * (alpha - 0.5)))
        elif interpolation_method == "radial":
            # Create a radial gradient centered in the mask
            height, width = mask.shape
            X, Y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
            distance = np.sqrt(X**2 + Y**2)
            weight = np.clip(1 - distance / np.sqrt(2), 0, 1) * alpha
        elif interpolation_method == "distance_transform":
            # Use distance transform on mask to calculate weights
            distance = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
            max_dist = distance.max() if distance.max() != 0 else 1.0
            weight = (1 - distance / max_dist) * alpha
        elif interpolation_method == "random_noise":
            # Use random noise as weight
            random_noise = np.random.rand(*mask.shape)
            weight = random_noise * alpha
        else:
            weight = alpha

        # Apply blending modes
        if blend_mode == "normal":
            interpolated_mask = (1 - weight) * mask + weight * mask_b_frame
        elif blend_mode == "add":
            interpolated_mask = np.clip(mask + mask_b_frame * weight, 0, 1)
        elif blend_mode == "multiply":
            interpolated_mask = mask * (mask_b_frame * weight + (1 - weight) * 1)
        elif blend_mode == "overlay":
            overlay = np.where(mask < 0.5, 2 * mask * (mask_b_frame * weight), 1 - 2 * (1 - mask) * (1 - mask_b_frame * weight))
            interpolated_mask = overlay
        elif blend_mode == "soft_light":
            soft_light = (1 - (1 - mask) * (1 - mask_b_frame * weight))
            interpolated_mask = soft_light
        else:
            interpolated_mask = (1 - weight) * mask + weight * mask_b_frame

        interpolated_mask = np.clip(interpolated_mask, 0.0, 1.0)
        return interpolated_mask.astype(np.float32)

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold,
                      invert, subtract_original, grow_with_blur, mask_b, 
                      interpolation_method, invert_mask_b, blend_mode, **kwargs):
        return (self.apply_mask_operation(masks, feature, feature_pipe, strength,
                                          feature_threshold, invert, subtract_original,
                                          grow_with_blur, mask_b=mask_b,
                                          interpolation_method=interpolation_method,
                                          invert_mask_b=invert_mask_b,
                                          blend_mode=blend_mode, **kwargs),)