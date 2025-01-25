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
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "morph_type": (["erode", "dilate", "open", "close"],),
            "max_kernel_size": ("INT", {"default": 5, "min": 3, "max": 21, "step": 2}),
            "max_iterations": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["max_kernel_size", "max_iterations", "None"]

    def apply_effect_internal(self, mask: np.ndarray, morph_type: str, **kwargs) -> np.ndarray:
        # Get values from kwargs - they'll have the same names as defined in INPUT_TYPES
        kernel_size = kwargs['max_kernel_size']
        iterations = kwargs['max_iterations']

        # Ensure kernel_size is odd and >= 3
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Ensure iterations is >= 1
        iterations = max(1, int(iterations))

        return morph_mask(mask, morph_type, kernel_size, iterations)

@apply_tooltips
class FlexMaskWarp(FlexMaskBase):
    #TODO: check warp functions for efficacy.....
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "warp_type": (["perlin", "radial", "swirl"],),


            "frequency": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            "max_amplitude": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 500.0, "step": 0.1}),
            "octaves": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["max_amplitude", "frequency", "octaves", "None"]

    def apply_effect_internal(self, mask: np.ndarray, warp_type: str, **kwargs) -> np.ndarray:
        # Get values from kwargs - they'll have the same names as defined in INPUT_TYPES
        amplitude = kwargs['max_amplitude']
        frequency = kwargs['frequency']
        octaves = kwargs['octaves']
        
        return warp_mask(mask, warp_type, frequency, amplitude, octaves)

@apply_tooltips
class FlexMaskTransform(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "transform_type": (["translate", "rotate", "scale"],),
            "max_x_value": ("FLOAT", {"default": 10.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
            "max_y_value": ("FLOAT", {"default": 10.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["max_x_value", "max_y_value", "None"]

    def apply_effect_internal(self, mask: np.ndarray, transform_type: str, **kwargs) -> np.ndarray:
        # Get values from kwargs - they'll have the same names as defined in INPUT_TYPES
        x_value = kwargs['max_x_value']
        y_value = kwargs['max_y_value']
        
        return transform_mask(mask, transform_type, x_value, y_value)

@apply_tooltips
class FlexMaskMath(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "mask_b": ("MASK",),
            "combination_method": (["add", "subtract", "multiply", "minimum", "maximum"],),
            "max_blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["max_blend", "None"]

    def apply_effect_internal(self, mask: np.ndarray, mask_b: torch.Tensor, combination_method: str, **kwargs) -> np.ndarray:
        # Get the frame index and handle mask_b indexing
        frame_index = kwargs.get('frame_index', 0)
        mask_b = mask_b[frame_index].numpy()
        
        # Get value from kwargs - it'll have the same name as defined in INPUT_TYPES
        blend = kwargs['max_blend']
        
        return combine_masks(mask, mask_b, combination_method, blend)

@apply_tooltips
class FlexMaskOpacity(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "max_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["max_opacity", "None"]

    def apply_effect_internal(self, mask: np.ndarray, **kwargs) -> np.ndarray:
        # Get value from kwargs - it'll have the same name as defined in INPUT_TYPES
        opacity = kwargs['max_opacity']
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
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
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
            "formula": (list(cls.formulas.keys()),),
            "a": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            "b": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["scale", "detail", "randomness", "seed", "x_offset", "y_offset", "None"]

    def generate_schedule(self, formula, feature_value, a, b):
        t = feature_value
        return self.formulas[formula](t, a, b)

    def apply_effect_internal(self, mask: np.ndarray, distance_metric: str, formula: str, a: float, b: float, feature_value: float, **kwargs) -> np.ndarray:
        height, width = mask.shape[:2]

        # Generate schedule value for modulation
        schedule_value = self.generate_schedule(formula, feature_value, a, b)

        # Get the pre-processed values from kwargs
        scale = kwargs['scale']
        detail = max(10, int(kwargs['detail']))  # Ensure detail is at least 10 and an integer
        randomness = max(0.0, kwargs['randomness'])  # Ensure randomness is non-negative
        seed = int(kwargs['seed'])
        x_offset = kwargs['x_offset']
        y_offset = kwargs['y_offset']

        # Create VoronoiNoise instance with parameters
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
        voronoi_mask = voronoi_tensor[0, :, :, 0].cpu().numpy()

        return voronoi_mask

@apply_tooltips
class FlexMaskBinary(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["threshold", "None"]

    def apply_effect_internal(self, mask: np.ndarray, **kwargs) -> np.ndarray:
        # Get the pre-processed threshold value from kwargs
        threshold = kwargs['threshold']
        return (mask > threshold).astype(np.float32)

#TODO: stateful node: make reset of state consistent, make state update pattern consistent, consistant state initialization in init
@apply_tooltips
class FlexMaskWavePropagation(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "wave_speed": ("FLOAT", {"default": 50.0, "min": 0.1, "max": 100.0, "step": 0.5}),
            "wave_amplitude": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05}),
            "wave_decay": ("FLOAT", {"default": 5.0, "min": 0.9, "max": 10.0, "step": 0.001}),
            "wave_frequency": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 10.0, "step": 0.01}),
            "max_wave_field": ("FLOAT", {"default": 750.0, "min": 10.0, "max": 10000.0, "step": 10.0}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["wave_speed", "wave_amplitude", "wave_decay", "wave_frequency", "max_wave_field", "None"]


    def __init__(self):
        super().__init__()
        self.wave_field = None
        self.frame_count = 0

    def process_below_threshold(self, mask, feature_value, strength, **kwargs):
        """Reset wave field when below threshold"""
        self.wave_field = None
        self.frame_count = 0
        return mask

    def apply_effect_internal(self, mask: np.ndarray, **kwargs) -> np.ndarray:
        # Get pre-processed values from kwargs
        max_wave_field = kwargs['max_wave_field']
        wave_speed = kwargs['wave_speed']
        wave_amplitude = kwargs['wave_amplitude']
        wave_decay = kwargs['wave_decay']
        wave_frequency = kwargs['wave_frequency']


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
            source_amplitude = wave_amplitude * np.sin(2 * np.pi * wave_frequency * self.frame_count * dt)
            self.wave_field[y, x] = source_amplitude

        # Propagate waves
        new_field = np.zeros_like(self.wave_field)
        for y in range(1, self.wave_field.shape[0] - 1):
            for x in range(1, self.wave_field.shape[1] - 1):
                # Simple wave equation discretization
                laplacian = (self.wave_field[y+1, x] + self.wave_field[y-1, x] + 
                           self.wave_field[y, x+1] + self.wave_field[y, x-1] - 
                           4 * self.wave_field[y, x])
                new_field[y, x] = self.wave_field[y, x] + wave_speed * dt * laplacian

        # Apply decay
        new_field *= np.exp(-wave_decay * dt)

        # Update wave field
        self.wave_field = new_field

        # Normalize and clip
        result = np.clip(self.wave_field / max_wave_field + mask, 0, 1)
        return result.astype(np.float32)

#TODO: stateful node: make reset of state consistent, make state update pattern consistent, consistant state initialization in init
#TODO: FIX THIS #IMPORTANT
@apply_tooltips
class FlexMaskEmanatingRings(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "ring_speed": ("FLOAT", {"default": 0.05, "min": 0.01, "max": 0.2, "step": 0.01}),
            "ring_width": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 0.5, "step": 0.01}),
            "ring_falloff": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "binary_mode": ("BOOLEAN", {"default": False}),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        return ["ring_speed", "ring_width", "ring_falloff", "None"]


    def __init__(self):
        super().__init__()
        self.rings = []
        # Cache the distance transform result
        self.last_mask = None
        self.cached_distance = None
        self.cached_max_distance = None
        
    def process_below_threshold(self, mask: np.ndarray, **kwargs) -> np.ndarray:
        kwargs['feature_value'] = 0
        return self.apply_effect_internal(mask, **kwargs)
        
    def apply_effect_internal(self, mask: np.ndarray, **kwargs) -> np.ndarray:
        # Get processed parameters
        ring_speed = kwargs['ring_speed']
        ring_width = kwargs['ring_width']
        ring_falloff = kwargs['ring_falloff']
        feature_value = kwargs.get('feature_value', 0)
        binary_mode = kwargs.get('binary_mode', False)
        
        # Use cached distance transform if mask hasn't changed
        if self.last_mask is None or not np.array_equal(mask, self.last_mask):
            distance = distance_transform_edt(1 - mask)
            max_distance = np.max(distance)
            if max_distance > 0:
                normalized_distance = distance / max_distance
            else:
                normalized_distance = distance
            # Cache results
            self.last_mask = mask.copy()
            self.cached_distance = normalized_distance
            self.cached_max_distance = max_distance
        else:
            normalized_distance = self.cached_distance
            max_distance = self.cached_max_distance

        if max_distance == 0:
            return mask.copy()

        # Only spawn new ring if feature_value is non-zero
        if feature_value > 0:
            self.rings.append({
                'progress': 0.0,
                'speed': ring_speed,
                'width': ring_width,
                'intensity': feature_value,
                'birth_time': kwargs.get('frame_index', 0)
            })

        # Always start with the original mask
        result = mask.copy()
        new_rings = []

        # Process all rings at once using vectorized operations
        if binary_mode:
            # Binary mode - sharp rings without falloff
            for ring in self.rings:
                ring['progress'] += ring['speed']
                if ring['progress'] < 1.0:
                    # Create sharp ring boundaries
                    ring_outer = normalized_distance < ring['progress']
                    ring_inner = normalized_distance < (ring['progress'] - ring['width'])
                    ring_mask = np.logical_xor(ring_outer, ring_inner)
                    result = np.logical_or(result, ring_mask)
                    new_rings.append(ring)
        else:
            # Smooth mode - with falloff
            for ring in self.rings:
                ring['progress'] += ring['speed']
                if ring['progress'] < 1.0:
                    # Vectorized ring calculation
                    ring_center = normalized_distance - ring['progress']
                    ring_mask = np.exp(-np.square(ring_center/ring['width']) * 4)
                    fade = np.power(1.0 - ring['progress'], ring_falloff * 3)
                    ring_mask *= fade * ring['intensity']
                    result = np.maximum(result, ring_mask)
                    new_rings.append(ring)

        self.rings = new_rings
        return (result > 0.5 if binary_mode else result).astype(np.float32)

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
                "feature_param": (["None", "max_num_shapes", "max_shape_size", "appearance_duration", "disappearance_duration"],),
            }
        }

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["max_num_shapes", "max_shape_size", "appearance_duration", "disappearance_duration", "None"]

    def __init__(self):
        super().__init__()
        self.shapes = []
        self.frame_count = 0

    def apply_effect_internal(self, mask: np.ndarray, appearance_method: str, easing_function: str, shape_type: str, **kwargs) -> np.ndarray:
        height, width = mask.shape
        result_mask = mask.copy()

        # Get pre-processed values from kwargs
        num_shapes = max(1, int(kwargs['max_num_shapes']))
        shape_size = kwargs['max_shape_size']
        app_duration = max(1, int(kwargs['appearance_duration']))
        disapp_duration = max(1, int(kwargs['disappearance_duration']))

        # Remove completed shapes
        self.shapes = [shape for shape in self.shapes if shape['frame'] < shape['total_frames']]

        # Add new shapes if needed (when feature_value > 0 or feature_param is None)
        feature_value = kwargs.get('feature_value', 0)
        if feature_value > 0 or kwargs.get('feature_param') == "None":
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
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "depth_map": ("IMAGE",),
            "z_front": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "z_back": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "feature_mode": (["squeeze", "expand", "move_forward", "move_back"],),
        })
        return base_inputs

    @classmethod
    def get_modifiable_params(cls):
        """Return parameters that can be modulated by features"""
        return ["z_front", "z_back", "both", "None"]

    def apply_effect_internal(self, mask: np.ndarray, depth_map: torch.Tensor, feature_mode: str, z_front: float, z_back: float,  **kwargs) -> np.ndarray:
        frame_index = kwargs.get('frame_index', 0)
        depth_map_frame = depth_map[frame_index].cpu().numpy()
        depth_map_frame = depth_map_frame[:, :, 0]

        #initialize values
        z_front_val = z_front
        z_back_val = z_back


        feature_value = kwargs.get('feature_value', 0)
        strength = kwargs.get('strength', 1.0)
        feature_param = kwargs.get('feature_param', 'None')            
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
        else:
            z_front_val = z_front
            z_back_val = z_back
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
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        base_inputs["required"].update({
            "depth_map": ("IMAGE",),
            "z1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "z2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "feature_mode": (["squeeze", "expand"],),
        })
        return base_inputs

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

    def apply_effect_internal(self, mask: torch.Tensor, depth_map: torch.Tensor, z1: float, z2: float, **kwargs) -> torch.Tensor:
        frame_index = kwargs.get('frame_index', 0)
        depth_map_frame = depth_map[frame_index, :, :, 0]

        # Get feature parameters from kwargs
        feature_value = kwargs.get('feature_value', 0)
        strength = kwargs.get('strength', 1.0)
        feature_param = kwargs.get('feature_param', 'None')
        feature_mode = kwargs.get('feature_mode', 'squeeze')

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
#TODO: delete this or merge with Math, or make unique
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
        return ["max_blend", "None"]

    def apply_effect_internal(self, mask: np.ndarray, mask_b: torch.Tensor, interpolation_method: str, 
                            invert_mask_b: bool, blend_mode: str, **kwargs) -> np.ndarray:
        frame_index = kwargs.get('frame_index', 0)
        mask_b_frame = mask_b[frame_index].numpy()

        if invert_mask_b:
            mask_b_frame = 1.0 - mask_b_frame

        # Ensure masks are in the same shape
        if mask.shape != mask_b_frame.shape:
            mask_b_frame = cv2.resize(mask_b_frame, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Get pre-processed blend value from kwargs
        blend = kwargs['max_blend']

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