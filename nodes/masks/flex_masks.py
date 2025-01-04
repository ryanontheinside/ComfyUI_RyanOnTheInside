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
from ...tooltips import apply_tooltips

@apply_tooltips
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

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["kernel_size", "iterations", "None"]

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, morph_type: str, max_kernel_size: int, max_iterations: int, feature_param: str, feature_mode: str, **kwargs) -> np.ndarray:
        # If feature_param is None, ignore feature_value entirely
        if feature_param == "None":
            kernel_size = max_kernel_size
            iterations = max_iterations
        # Otherwise, handle parameter modulation based on selection
        elif feature_param == "kernel_size":
            kernel_size = self.modulate_param("kernel_size", max_kernel_size, feature_value, strength, feature_mode)
            kernel_size = max(3, int(kernel_size))  # Ensure odd number >= 3
            if kernel_size % 2 == 0:
                kernel_size += 1
            iterations = max_iterations
        elif feature_param == "iterations":
            kernel_size = max_kernel_size
            iterations = self.modulate_param("iterations", max_iterations, feature_value, strength, feature_mode)
            iterations = max(1, int(iterations))
        return morph_mask(mask, morph_type, kernel_size, iterations)

@apply_tooltips
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

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["amplitude", "frequency", "octaves", "None"]

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, warp_type: str, frequency: float, max_amplitude: float, octaves: int, feature_param: str, feature_mode: str, **kwargs) -> np.ndarray:
        # If feature_param is None, use direct values without modulation
        if feature_param == "None":
            amplitude = max_amplitude
            frequency_val = frequency
            octaves_val = octaves
        # Otherwise, handle parameter modulation based on selection
        elif feature_param == "amplitude":
            amplitude = self.modulate_param("amplitude", max_amplitude, feature_value, strength, feature_mode)
            frequency_val = frequency
            octaves_val = octaves
        elif feature_param == "frequency":
            amplitude = max_amplitude
            frequency_val = self.modulate_param("frequency", frequency, feature_value, strength, feature_mode)
            # Ensure frequency doesn't get too close to zero to prevent division issues
            frequency_val = max(0.01, frequency_val)  # Minimum frequency of 0.01
            octaves_val = octaves
        elif feature_param == "octaves":
            amplitude = max_amplitude
            frequency_val = frequency
            octaves_val = self.modulate_param("octaves", octaves, feature_value, strength, feature_mode)
            octaves_val = max(1, int(octaves_val))
        
        return warp_mask(mask, warp_type, frequency_val, amplitude, octaves_val)

@apply_tooltips
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

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["x_value", "y_value", "None"]

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, transform_type: str, max_x_value: float, max_y_value: float, feature_param: str, feature_mode: str, **kwargs) -> np.ndarray:
        # If feature_param is None, use direct values without modulation
        if feature_param == "None":
            x_value = max_x_value
            y_value = max_y_value
        # Otherwise, handle parameter modulation based on selection
        elif feature_param == "x_value":
            x_value = self.modulate_param("x_value", max_x_value, feature_value, strength, feature_mode)
            y_value = max_y_value
        elif feature_param == "y_value":
            x_value = max_x_value
            y_value = self.modulate_param("y_value", max_y_value, feature_value, strength, feature_mode)
        
        return transform_mask(mask, transform_type, x_value, y_value)

@apply_tooltips
class FlexMaskMath(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "mask_b": ("MASK",),
                "combination_method": (["add", "subtract", "multiply", "minimum", "maximum"],),
                "max_blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["blend", "None"]

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, mask_b: torch.Tensor, combination_method: str, max_blend: float, feature_param: str, feature_mode: str, **kwargs) -> np.ndarray:
        frame_index = kwargs.get('frame_index', 0)
        mask_b = mask_b[frame_index].numpy()
        
        # If feature_param is None, use direct values without modulation
        if feature_param == "None":
            blend = max_blend
        # Otherwise, handle parameter modulation based on selection
        elif feature_param == "blend":
            blend = self.modulate_param("blend", max_blend, feature_value, strength, feature_mode)
            blend = np.clip(blend, 0.0, 1.0)  # Ensure blend stays in valid range
        
        return combine_masks(mask, mask_b, combination_method, blend)

@apply_tooltips
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

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["opacity", "None"]

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, max_opacity: float, feature_param: str, feature_mode: str, **kwargs) -> np.ndarray:
        # If feature_param is None, use direct values without modulation
        if feature_param == "None":
            opacity = max_opacity
        # Otherwise, handle parameter modulation based on selection
        elif feature_param == "opacity":
            opacity = self.modulate_param("opacity", max_opacity, feature_value, strength, feature_mode)
            opacity = np.clip(opacity, 0.0, 1.0)  # Ensure opacity stays in valid range
        
        return mask * opacity

@apply_tooltips
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
                "feature_param": (["None", "scale", "detail", "randomness", "seed", "x_offset", "y_offset"],),
                "formula": (list(cls.formulas.keys()),),
                "a": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "b": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["scale", "detail", "randomness", "seed", "x_offset", "y_offset", "None"]

    def generate_schedule(self, formula, feature_value, a, b):
        t = feature_value
        return self.formulas[formula](t, a, b)

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
                     distance_metric: str, scale: float, detail: int, randomness: float, 
                     seed: int, x_offset: float, y_offset: float, feature_param: str,
                     formula: str, a: float, b: float, feature_mode: str, **kwargs) -> np.ndarray:
        
        height, width = mask.shape[:2]

        # Generate schedule value for modulation
        schedule_value = self.generate_schedule(formula, feature_value, a, b)

        # If feature_param is None, use direct values without modulation
        if feature_param == "None":
            scale_val = scale
            detail_val = detail
            randomness_val = randomness
            seed_val = seed
            x_offset_val = x_offset
            y_offset_val = y_offset
        else:
            # Initialize with base values
            scale_val = scale
            detail_val = detail
            randomness_val = randomness
            seed_val = seed
            x_offset_val = x_offset
            y_offset_val = y_offset

            # Apply modulation based on feature_param
            if feature_param == "scale":
                scale_val = self.modulate_param("scale", scale, schedule_value, strength, feature_mode)
            elif feature_param == "detail":
                detail_val = self.modulate_param("detail", detail, schedule_value, strength, feature_mode)
                detail_val = max(10, int(detail_val))
            elif feature_param == "randomness":
                randomness_val = self.modulate_param("randomness", randomness, schedule_value, strength, feature_mode)
                randomness_val = max(0.0, randomness_val)
            elif feature_param == "seed":
                seed_val = int(seed + (schedule_value * strength * 1000000))
            elif feature_param == "x_offset":
                x_offset_val = self.modulate_param("x_offset", x_offset, schedule_value, strength, feature_mode)
            elif feature_param == "y_offset":
                y_offset_val = self.modulate_param("y_offset", y_offset, schedule_value, strength, feature_mode)

        # Create VoronoiNoise instance with modulated parameters
        voronoi = VoronoiNoise(
            width=width, 
            height=height, 
            scale=[scale_val], 
            detail=[detail_val], 
            seed=[seed_val], 
            randomness=[randomness_val],
            X=[x_offset_val],
            Y=[y_offset_val],
            distance_metric=distance_metric,
            batch_size=1,
            device=get_torch_device()
        )

        # Generate Voronoi noise
        voronoi_tensor = voronoi()
        voronoi_mask = voronoi_tensor[0, :, :, 0].cpu().numpy()

        return voronoi_mask

@apply_tooltips
class FlexMaskBinary(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["threshold", "None"]

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, threshold: float, **kwargs) -> np.ndarray:
        # Modulate the threshold based on the feature value and strength
        modulated_threshold = self.modulate_param("threshold", threshold, feature_value, strength, kwargs.get("feature_mode", "relative"))
        return (mask > modulated_threshold).astype(np.float32)

#TODO: stateful node: make reset of state consistent, make state update pattern consistent, consistant state initialization in init
@apply_tooltips
class FlexMaskWavePropagation(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
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

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["wave_speed", "wave_amplitude", "wave_decay", "wave_frequency", "None"]

    def __init__(self):
        super().__init__()
        self.wave_field = None
        self.frame_count = 0

    def process_mask_below_threshold(self, mask, feature_value, strength, **kwargs):
        """Reset wave field when below threshold"""
        self.wave_field = None
        self.frame_count = 0
        return mask

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
                     wave_speed: float, wave_amplitude: float, wave_decay: float, 
                     wave_frequency: float, max_wave_field: float, feature_param: str, 
                     feature_mode: str, **kwargs) -> np.ndarray:
        # If feature_param is None, use direct values without modulation
        if feature_param == "None":
            speed = wave_speed
            amplitude = wave_amplitude
            decay = wave_decay
            frequency = wave_frequency
        else:
            # Apply modulation based on feature_param
            if feature_param == "wave_speed":
                speed = self.modulate_param("wave_speed", wave_speed, feature_value, strength, feature_mode)
                amplitude = wave_amplitude
                decay = wave_decay
                frequency = wave_frequency
            elif feature_param == "wave_amplitude":
                speed = wave_speed
                amplitude = self.modulate_param("wave_amplitude", wave_amplitude, feature_value, strength, feature_mode)
                decay = wave_decay
                frequency = wave_frequency
            elif feature_param == "wave_decay":
                speed = wave_speed
                amplitude = wave_amplitude
                decay = self.modulate_param("wave_decay", wave_decay, feature_value, strength, feature_mode)
                frequency = wave_frequency
            elif feature_param == "wave_frequency":
                speed = wave_speed
                amplitude = wave_amplitude
                decay = wave_decay
                frequency = self.modulate_param("wave_frequency", wave_frequency, feature_value, strength, feature_mode)

        # Initialize wave field if needed
        if self.wave_field is None:
            self.wave_field = np.zeros_like(mask)
            self.frame_count = 0

        # Update wave field
        dt = 1.0 / 30.0  # Assuming 30 fps
        self.frame_count += 1

        # Create wave sources from mask
        wave_sources = np.where(mask > 0.5)
        for y, x in zip(*wave_sources):
            source_amplitude = amplitude * np.sin(2 * np.pi * frequency * self.frame_count * dt)
            self.wave_field[y, x] = source_amplitude

        # Propagate waves
        new_field = np.zeros_like(self.wave_field)
        for y in range(1, self.wave_field.shape[0] - 1):
            for x in range(1, self.wave_field.shape[1] - 1):
                # Simple wave equation discretization
                laplacian = (self.wave_field[y+1, x] + self.wave_field[y-1, x] + 
                           self.wave_field[y, x+1] + self.wave_field[y, x-1] - 
                           4 * self.wave_field[y, x])
                new_field[y, x] = self.wave_field[y, x] + speed * dt * laplacian

        # Apply decay
        new_field *= np.exp(-decay * dt)

        # Update wave field
        self.wave_field = new_field

        # Normalize and clip
        result = np.clip(self.wave_field / max_wave_field + mask, 0, 1)
        return result.astype(np.float32)

#TODO: stateful node: make reset of state consistent, make state update pattern consistent, consistant state initialization in init
@apply_tooltips
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
                "feature_param": (["None", "num_rings", "ring_width", "wave_speed", "all"],),
            }
        }

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["num_rings", "ring_width", "wave_speed", "all", "None"]

    def __init__(self):
        super().__init__()
        self.rings = []

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float,
                     num_rings: int, max_ring_width: float, wave_speed: float,
                     feature_param: str, feature_mode: str, **kwargs) -> np.ndarray:
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

        # If feature_param is None, use direct values without modulation
        if feature_param == "None":
            adjusted_num_rings = num_rings
            adjusted_max_ring_width = max_ring_width
            adjusted_wave_speed = wave_speed
        else:
            # Apply modulation based on feature_param
            if feature_param in ["num_rings", "all"]:
                adjusted_num_rings = self.modulate_param("num_rings", num_rings, feature_value, strength, feature_mode)
                adjusted_num_rings = max(1, int(adjusted_num_rings))
            else:
                adjusted_num_rings = num_rings

            if feature_param in ["ring_width", "all"]:
                adjusted_max_ring_width = self.modulate_param("ring_width", max_ring_width, feature_value, strength, feature_mode)
                adjusted_max_ring_width = np.clip(adjusted_max_ring_width, 0.01, 0.9)
            else:
                adjusted_max_ring_width = max_ring_width

            if feature_param in ["wave_speed", "all"]:
                adjusted_wave_speed = self.modulate_param("wave_speed", wave_speed, feature_value, strength, feature_mode)
                adjusted_wave_speed = np.clip(adjusted_wave_speed, 0.01, 0.5)
            else:
                adjusted_wave_speed = wave_speed

        # Create new rings if feature_value > 0 or feature_param is None
        if feature_value > 0 or feature_param == "None":
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

#TODO: stateful node: make reset of state consistent, make state update pattern consistent, consistant state initialization in init
@apply_tooltips
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
                "feature_param": (["None", "num_shapes", "shape_size", "appearance_duration", "disappearance_duration"],),
            }
        }

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["num_shapes", "shape_size", "appearance_duration", "disappearance_duration", "None"]

    def __init__(self):
        super().__init__()
        self.shapes = []
        self.frame_count = 0

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float,
                     max_num_shapes: int, max_shape_size: float, appearance_duration: int,
                     disappearance_duration: int, appearance_method: str, easing_function: str,
                     shape_type: str, feature_param: str, feature_mode: str, **kwargs) -> np.ndarray:
        height, width = mask.shape
        result_mask = mask.copy()

        # If feature_param is None, use direct values without modulation
        if feature_param == "None":
            num_shapes = max_num_shapes
            shape_size = max_shape_size
            app_duration = appearance_duration
            disapp_duration = disappearance_duration
        else:
            # Apply modulation based on feature_param
            if feature_param == "num_shapes":
                num_shapes = self.modulate_param("num_shapes", max_num_shapes, feature_value, strength, feature_mode)
                num_shapes = max(1, int(num_shapes))
                shape_size = max_shape_size
                app_duration = appearance_duration
                disapp_duration = disappearance_duration
            elif feature_param == "shape_size":
                num_shapes = max_num_shapes
                shape_size = self.modulate_param("shape_size", max_shape_size, feature_value, strength, feature_mode)
                shape_size = np.clip(shape_size, 0.01, 1.0)
                app_duration = appearance_duration
                disapp_duration = disappearance_duration
            elif feature_param == "appearance_duration":
                num_shapes = max_num_shapes
                shape_size = max_shape_size
                app_duration = self.modulate_param("appearance_duration", appearance_duration, feature_value, strength, feature_mode)
                app_duration = max(1, int(app_duration))
                disapp_duration = disappearance_duration
            elif feature_param == "disappearance_duration":
                num_shapes = max_num_shapes
                shape_size = max_shape_size
                app_duration = appearance_duration
                disapp_duration = self.modulate_param("disappearance_duration", disappearance_duration, feature_value, strength, feature_mode)
                disapp_duration = max(1, int(disapp_duration))

        # Remove completed shapes
        self.shapes = [shape for shape in self.shapes if shape['frame'] < shape['total_frames']]

        # Add new shapes if needed (when feature_value > 0 or feature_param is None)
        if feature_value > 0 or feature_param == "None":
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

@apply_tooltips
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

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["z_front", "z_back", "both", "None"]

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
                     depth_map: torch.Tensor, z_front: float, z_back: float, feature_param: str, 
                     feature_mode: str, **kwargs) -> np.ndarray:
        frame_index = kwargs.get('frame_index', 0)
        depth_map_frame = depth_map[frame_index].cpu().numpy()
        depth_map_frame = depth_map_frame[:, :, 0]

        # If feature_param is None, use direct values without modulation
        if feature_param == "None":
            z_front_val = z_front
            z_back_val = z_back
        else:
            z_front_val = z_front
            z_back_val = z_back
            
            # Apply modulation based on feature_mode and feature_param
            if feature_mode == "squeeze":
                if feature_param in ["z_front", "both"]:
                    z_front_val = z_front - (z_front - z_back) * strength * feature_value / 2 if z_front > z_back else z_front + (z_back - z_front) * strength * feature_value / 2
                if feature_param in ["z_back", "both"]:
                    z_back_val = z_back + (z_front - z_back) * strength * feature_value / 2 if z_back < z_front else z_back - (z_back - z_front) * strength * feature_value / 2
            elif feature_mode == "expand":
                if feature_param in ["z_front", "both"]:
                    z_front_val = min(1.0, z_front + (z_front - z_back) * strength * feature_value / 2) if z_front > z_back else max(0.0, z_front - (z_back - z_front) * strength * feature_value / 2)
                if feature_param in ["z_back", "both"]:
                    z_back_val = max(0.0, z_back - (z_front - z_back) * strength * feature_value / 2) if z_back < z_front else min(1.0, z_back + (z_back - z_front) * strength * feature_value / 2)
            elif feature_mode == "move_forward":
                if feature_param in ["z_front", "both"]:
                    z_front_val = min(1.0, z_front + strength * feature_value)
                if feature_param in ["z_back", "both"]:
                    z_back_val = min(1.0, z_back + strength * feature_value)
            elif feature_mode == "move_back":
                if feature_param in ["z_front", "both"]:
                    z_front_val = max(0.0, z_front - strength * feature_value)
                if feature_param in ["z_back", "both"]:
                    z_back_val = max(0.0, z_back - strength * feature_value)

        # Create the depth mask
        if z_back_val < z_front_val:
            depth_mask = (depth_map_frame >= z_back_val) & (depth_map_frame <= z_front_val)
        else:
            depth_mask = (depth_map_frame >= z_back_val) | (depth_map_frame <= z_front_val)

        depth_mask_resized = cv2.resize(depth_mask.astype(np.float32), (mask.shape[1], mask.shape[0]))

        # Subtract anything that doesn't fall within the input mask
        combined_mask = np.where(mask > 0, depth_mask_resized, 0)

        return combined_mask

@apply_tooltips
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
                "feature_param": (["None", "z1", "z2", "both"],),
                "feature_mode": (["squeeze", "expand"],),
            }
        }

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["z1", "z2", "both", "None"]

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
        return torch.median(torch.tensor(sizes)).item()

    def process_mask(self, mask: torch.Tensor, feature_value: float, strength: float, 
                     depth_map: torch.Tensor, z1: float, z2: float, feature_param: str, 
                     feature_mode: str, **kwargs) -> torch.Tensor:
        frame_index = kwargs.get('frame_index', 0)
        depth_map_frame = depth_map[frame_index, :, :, 0]

        # Calculate the ROI size for the current frame
        roi_size = self.calculate_roi_size(mask)
        reference_size = kwargs.get('reference_size', roi_size)  # Use current frame as reference if not provided

        # If feature_param is None, use direct values without modulation
        if feature_param == "None":
            z1_val = z1
            z2_val = z2
        else:
            # Apply modulation based on feature_param
            if feature_param in ["z1", "both"]:
                z1_val = self.modulate_param("z1", z1, feature_value, strength, feature_mode)
                z1_val = z1_val * (roi_size / reference_size)
            else:
                z1_val = z1 * (roi_size / reference_size)

            if feature_param in ["z2", "both"]:
                z2_val = self.modulate_param("z2", z2, feature_value, strength, feature_mode)
                z2_val = z2_val * (roi_size / reference_size)
            else:
                z2_val = z2 * (roi_size / reference_size)

        # Ensure z1 is less than z2
        z1_val, z2_val = min(z1_val, z2_val), max(z1_val, z2_val)

        # Apply depth masking based on feature_mode
        if feature_mode == "squeeze":
            depth_mask = (depth_map_frame >= z1_val) & (depth_map_frame <= z2_val)
        elif feature_mode == "expand":
            depth_mask = (depth_map_frame < z1_val) | (depth_map_frame > z2_val)

        # Resize and combine with input mask
        depth_mask_resized = F.interpolate(depth_mask.unsqueeze(0).unsqueeze(0).float(), 
                                         size=mask.shape[-2:], 
                                         mode='nearest').squeeze(0).squeeze(0)
        output_mask = mask.float() * depth_mask_resized

        return output_mask

@apply_tooltips
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
                "max_blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert_mask_b": ("BOOLEAN", {"default": False}),
                "blend_mode": (["normal", "add", "multiply", "overlay", "soft_light"],),
            }
        }

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["blend", "None"]

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
                     mask_b: torch.Tensor, interpolation_method: str, max_blend: float,
                     invert_mask_b: bool, blend_mode: str, feature_param: str, 
                     feature_mode: str, **kwargs) -> np.ndarray:
        frame_index = kwargs.get('frame_index', 0)
        mask_b_frame = mask_b[frame_index].numpy()

        if invert_mask_b:
            mask_b_frame = 1.0 - mask_b_frame

        # Ensure masks are in the same shape
        if mask.shape != mask_b_frame.shape:
            mask_b_frame = cv2.resize(mask_b_frame, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Handle blend parameter based on feature_param
        if feature_param == "None":
            blend = max_blend
        elif feature_param == "blend":
            blend = self.modulate_param("blend", max_blend, feature_value, strength, feature_mode)
            blend = np.clip(blend, 0.0, 1.0)  # Ensure blend stays in valid range

        # Apply interpolation method to compute weight
        if interpolation_method == "linear":
            weight = blend
        elif interpolation_method == "ease_in":
            weight = blend ** 2
        elif interpolation_method == "ease_out":
            weight = 1 - (1 - blend) ** 2
        elif interpolation_method == "ease_in_out":
            weight = blend ** 2 / (blend ** 2 + (1 - blend) ** 2 + 1e-6)
        elif interpolation_method == "cubic":
            weight = 3 * blend ** 2 - 2 * blend ** 3
        elif interpolation_method == "sigmoid":
            weight = 1 / (1 + np.exp(-12 * (blend - 0.5)))
        elif interpolation_method == "radial":
            # Create a radial gradient centered in the mask
            height, width = mask.shape
            X, Y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
            distance = np.sqrt(X**2 + Y**2)
            weight = np.clip(1 - distance / np.sqrt(2), 0, 1) * blend
        elif interpolation_method == "distance_transform":
            # Use distance transform on mask to calculate weights
            distance = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
            max_dist = distance.max() if distance.max() != 0 else 1.0
            weight = (1 - distance / max_dist) * blend
        elif interpolation_method == "random_noise":
            # Use random noise as weight
            random_noise = np.random.rand(*mask.shape)
            weight = random_noise * blend
        else:
            weight = blend

        # Apply blending modes
        if blend_mode == "normal":
            interpolated_mask = (1 - weight) * mask + weight * mask_b_frame
        elif blend_mode == "add":
            interpolated_mask = np.clip(mask + mask_b_frame * weight, 0, 1)
        elif blend_mode == "multiply":
            interpolated_mask = mask * (mask_b_frame * weight + (1 - weight) * 1)
        elif blend_mode == "overlay":
            overlay = np.where(mask < 0.5, 
                             2 * mask * (mask_b_frame * weight), 
                             1 - 2 * (1 - mask) * (1 - mask_b_frame * weight))
            interpolated_mask = overlay
        elif blend_mode == "soft_light":
            soft_light = (1 - (1 - mask) * (1 - mask_b_frame * weight))
            interpolated_mask = soft_light
        else:
            interpolated_mask = (1 - weight) * mask + weight * mask_b_frame

        return np.clip(interpolated_mask, 0.0, 1.0).astype(np.float32)