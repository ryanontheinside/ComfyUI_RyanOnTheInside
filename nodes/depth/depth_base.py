import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from comfy.utils import ProgressBar
from ... import RyanOnTheInside
import cv2

class FlexDepthBase(RyanOnTheInside, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_maps": ("IMAGE",),
                "feature": ("FEATURE",),
                "feature_pipe": ("FEATURE_PIPE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_param": (cls.get_modifiable_params(), {"default": cls.get_modifiable_params()[0]}),
                "feature_mode": (["relative", "absolute"], {"default": "relative"}),
            }
        }

    CATEGORY = "RyanOnTheInside/DepthEffects"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"

    def __init__(self):
        self.progress_bar = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None

    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        return []

    def modulate_param(self, param_name, param_value, feature_value, strength, mode):
        if mode == "relative":
            return param_value * (1 + (feature_value - 0.5) * strength)
        else:  # absolute
            return param_value * feature_value * strength

    def apply_effect(self, depth_maps, feature, feature_pipe, strength, feature_threshold, feature_param, feature_mode, **kwargs):
        num_frames = feature_pipe.frame_count
        depth_maps_np = depth_maps.cpu().numpy()

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        result = []
        for i in range(num_frames):
            depth_map = depth_maps_np[i]
            feature_value = feature.get_value_at_frame(i)
            kwargs['frame_index'] = i
            if feature_value >= feature_threshold:
                processed_depth_map = self.process_depth_map(depth_map, feature_value, strength, 
                                                             feature_param=feature_param, 
                                                             feature_mode=feature_mode, 
                                                             **kwargs)
            else:
                processed_depth_map = depth_map

            result.append(processed_depth_map)
            self.update_progress()

        self.end_progress()

        # Convert the list of numpy arrays to a single numpy array
        result_np = np.stack(result)
        
        # Convert the numpy array to a PyTorch tensor
        result_tensor = torch.from_numpy(result_np).float()

        # Ensure the tensor is in BHWC format
        if result_tensor.shape[1] != depth_maps.shape[1]:  # Adjust as needed
            result_tensor = result_tensor.permute(0, 2, 3, 1)

        return (result_tensor,)

    def process_depth_map(self, depth_map: np.ndarray, feature_value: float, strength: float, 
                          feature_param: str, feature_mode: str, **kwargs) -> np.ndarray:
        # Modulate the selected parameter
        for param_name in self.get_modifiable_params():
            if param_name in kwargs:
                if param_name == feature_param:
                    kwargs[param_name] = self.modulate_param(param_name, kwargs[param_name], 
                                                             feature_value, strength, feature_mode)
        
        # Call the child class's implementation
        return self.apply_effect_internal(depth_map, **kwargs)

    @abstractmethod
    def apply_effect_internal(self, depth_map: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect to the depth map. To be implemented by child classes."""
        pass


class DepthInjection(FlexDepthBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "mask": ("MASK",),
                "gradient_steepness": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "depth_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    CATEGORY = "RyanOnTheInside/DepthModifiers"

    @classmethod
    def get_modifiable_params(cls):
        return ["gradient_steepness", "depth_min", "depth_max", "strength", "None"]

    def apply_effect_internal(self, depth_map: np.ndarray, mask, gradient_steepness, depth_min, depth_max, frame_index, **kwargs) -> np.ndarray:
        h, w, c = depth_map.shape

        # Get mask for this frame
        mask_i = mask[frame_index].cpu().numpy().astype(np.uint8)

        # Use cv2 to find contours (separate shapes)
        contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sphere_gradients = np.zeros((h, w), dtype=np.float32)

        for contour in contours:
            # Create a mask for this contour
            component_mask = np.zeros_like(mask_i)
            cv2.drawContours(component_mask, [contour], 0, 1, -1)

            # Find center and radius of the circular component
            M = cv2.moments(contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
            else:
                continue  # Skip this contour if we can't find its center

            # Calculate radius (approximate)
            radius = np.sqrt(cv2.contourArea(contour) / np.pi)

            # Generate spherical gradient
            y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            distances = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            sphere_gradient = np.clip(1 - (distances / radius) ** gradient_steepness, 0, 1)

            # Apply gradient only to this component
            sphere_gradients += sphere_gradient * component_mask.astype(np.float32)

        # Scale gradient to depth range
        depth_gradient = depth_min + sphere_gradients * (depth_max - depth_min)

        # Apply gradient to depth map
        modified_depth = depth_map.copy()
        depth_gradient = np.expand_dims(depth_gradient, axis=-1).repeat(c, axis=-1)
        mask_i_expanded = np.expand_dims(mask_i, axis=-1).repeat(c, axis=-1).astype(bool)
        modified_depth = np.where(mask_i_expanded,
                                  depth_gradient,
                                  modified_depth)

        # Blend modified depth with original depth
        strength = kwargs.get('strength', 1.0)
        blend_mask = mask_i_expanded * strength
        modified_depth = depth_map * (1 - blend_mask) + modified_depth * blend_mask

        return modified_depth
    
class DepthBlender(FlexDepthBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "other_depth_maps": ("IMAGE",),
                "blend_mode": (["add", "subtract", "multiply", "average"], {"default": "average"}),
                # "strength" is already included in the base class
            }
        }

    CATEGORY = "RyanOnTheInside/DepthModifiers"

    @classmethod
    def get_modifiable_params(cls):
        return ["strength", "None"]

    def apply_effect_internal(self, depth_map: np.ndarray, other_depth_maps, frame_index, **kwargs) -> np.ndarray:
        blend_mode = kwargs.get('blend_mode')
        strength = kwargs.get('strength', 1.0)

        # Get the other depth map for the current frame
        other_depth_map = other_depth_maps[frame_index].cpu().numpy()

        if blend_mode == "add":
            blended_depth = depth_map + other_depth_map
        elif blend_mode == "subtract":
            blended_depth = depth_map - other_depth_map
        elif blend_mode == "multiply":
            blended_depth = depth_map * other_depth_map
        elif blend_mode == "average":
            blended_depth = (depth_map + other_depth_map) / 2
        else:
            blended_depth = depth_map  # Default to original depth map

        # Blend with the original depth map based on strength
        modified_depth = depth_map * (1 - strength) + blended_depth * strength

        # Ensure depth values are within valid range
        modified_depth = np.clip(modified_depth, 0.0, 1.0)

        return modified_depth
    

class DepthRippleEffect(FlexDepthBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "ripple_amplitude": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                "ripple_frequency": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "ripple_phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 6.2832, "step": 0.1}),  # 2π
                # "strength" is already included in the base class
            }
        }

    CATEGORY = "RyanOnTheInside/DepthModifiers"

    @classmethod
    def get_modifiable_params(cls):
        return ["ripple_amplitude", "ripple_frequency", "ripple_phase", "strength", "None"]

    def apply_effect_internal(self, depth_map: np.ndarray, frame_index, **kwargs) -> np.ndarray:
        ripple_amplitude = kwargs.get('ripple_amplitude')
        ripple_frequency = kwargs.get('ripple_frequency')
        ripple_phase = kwargs.get('ripple_phase')
        strength = kwargs.get('strength', 1.0)

        h, w, c = depth_map.shape

        # Create a coordinate grid
        y, x = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')

        # Compute ripple pattern
        ripple = ripple_amplitude * np.sin(2 * np.pi * ripple_frequency * (x + y) + ripple_phase)

        # Apply ripple to depth map
        ripple = np.expand_dims(ripple, axis=-1).repeat(c, axis=-1)
        modified_depth = depth_map + ripple

        # Blend with the original depth map based on strength
        modified_depth = depth_map * (1 - strength) + modified_depth * strength

        # Ensure depth values are within valid range
        modified_depth = np.clip(modified_depth, 0.0, 1.0)

        return modified_depth