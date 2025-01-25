import torch
import numpy as np
from abc import ABC, abstractmethod
from comfy.utils import ProgressBar
from ..flex.flex_base import FlexBase
import cv2
from ...tooltips import apply_tooltips

#NOTE: in hindsight, much of this would have been better suited as mask-based operations
@apply_tooltips
class FlexDepthBase(FlexBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "depth_maps": ("IMAGE",),
            },
            # Optional inputs are inherited from FlexBase
        }

    CATEGORY = "RyanOnTheInside/DepthEffects"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"

    def __init__(self):
        super().__init__()

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

    def apply_effect(
        self,
        depth_maps,
        strength,
        feature_threshold,
        feature_param,
        feature_mode,
        opt_feature=None,
        **kwargs
    ):
        num_frames = depth_maps.shape[0]
        depth_maps_np = depth_maps.cpu().numpy()

        if opt_feature is None:
            self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

            result = []
            for i in range(num_frames):
                processed_depth_map = self.process_depth_map(
                    depth_maps_np[i],
                    0.5,  # Default feature value when no feature is provided
                    strength,
                    feature_param=feature_param,
                    feature_mode=feature_mode,
                    frame_index=i,
                    **kwargs
                )
                result.append(processed_depth_map)
                self.update_progress()
        else:
            num_frames = opt_feature.frame_count
            self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

            result = []
            for i in range(num_frames):
                depth_map = depth_maps_np[i]
                feature_value = self.get_feature_value(i, opt_feature)
                feature_value = 0.5 if feature_value is None else feature_value
                kwargs['frame_index'] = i
                if feature_value >= feature_threshold:
                    processed_depth_map = self.process_depth_map(
                        depth_map,
                        feature_value,
                        strength,
                        feature_param=feature_param,
                        feature_mode=feature_mode,
                        **kwargs
                    )
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
        if result_tensor.shape[1] != depth_maps.shape[1]:
            result_tensor = result_tensor.permute(0, 2, 3, 1)

        return (result_tensor,)

    def process_depth_map(
        self,
        depth_map: np.ndarray,
        feature_value: float,
        strength: float,
        feature_param: str,
        feature_mode: str,
        **kwargs
    ) -> np.ndarray:
        # Modulate the selected parameter
        for param_name in self.get_modifiable_params():
            if param_name in kwargs and param_name == feature_param:
                kwargs[param_name] = self.modulate_param(
                    param_name,
                    kwargs[param_name],
                    feature_value,
                    strength,
                    feature_mode
                )
        # Call the child class's implementation
        return self.apply_effect_internal(depth_map, **kwargs)

    @abstractmethod
    def apply_effect_internal(self, depth_map: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect to the depth map. To be implemented by child classes."""
        pass


@apply_tooltips
class DepthInjection(FlexDepthBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs["required"]
        
        # Update feature_param first
        base_required["feature_param"] = cls.get_modifiable_params()
        

        # Then add other inputs
        base_required.update({
            "mask": ("MASK",),
            "gradient_steepness": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            "depth_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "depth_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        
        return {
            "required": base_required,
            "optional": base_inputs["optional"],
        }

    CATEGORY = "RyanOnTheInside/DepthModifiers"

    @classmethod
    def get_modifiable_params(cls):
        return ["gradient_steepness", "depth_min", "depth_max", "strength", "None"]

    def apply_effect_internal(self, depth_map: np.ndarray, mask, frame_index, **kwargs) -> np.ndarray:
        gradient_steepness = kwargs.get('gradient_steepness')
        depth_min = kwargs.get('depth_min')
        depth_max = kwargs.get('depth_max')
        strength = kwargs.get('strength', 1.0)

        h, w, c = depth_map.shape

        # Get mask for this frame
        mask_i = mask[frame_index].cpu().numpy().astype(np.uint8)

        # Label connected components in the mask
        num_labels, labels = cv2.connectedComponents(mask_i)

        # Initialize the gradient map
        gradient_map = np.zeros((h, w), dtype=np.float32)

        for label in range(1, num_labels):  # Skip background label 0
            # Create a mask for this component
            component_mask = (labels == label).astype(np.uint8)

            # Compute distance transform inside the component
            dist_transform = cv2.distanceTransform(component_mask, cv2.DIST_L2, 5)
            max_dist = np.max(dist_transform)

            if max_dist > 0:
                # Normalize distance transform to [0,1]
                normalized_dist = dist_transform / max_dist

                # Apply gradient steepness
                shape_gradient = np.power(normalized_dist, gradient_steepness)

                # Apply gradient only to this component
                gradient_map += shape_gradient * component_mask.astype(np.float32)

        # Scale gradient to depth range
        depth_gradient = depth_min + gradient_map * (depth_max - depth_min)

        # Apply gradient to depth map
        modified_depth = depth_map.copy()
        depth_gradient_expanded = np.expand_dims(depth_gradient, axis=-1).repeat(c, axis=-1)
        mask_i_expanded = np.expand_dims(mask_i, axis=-1).repeat(c, axis=-1).astype(bool)
        modified_depth = np.where(mask_i_expanded,
                                  depth_gradient_expanded,
                                  modified_depth)

        # Blend modified depth with original depth
        blend_mask = mask_i_expanded * strength
        modified_depth = depth_map * (1 - blend_mask) + modified_depth * blend_mask

        return modified_depth

    
@apply_tooltips
class DepthBlender(FlexDepthBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs["required"]
        
        # Update feature_param first
        base_required["feature_param"] = cls.get_modifiable_params()
        

        # Then add other inputs
        base_required.update({
            "other_depth_maps": ("IMAGE",),
            "blend_mode": (["add", "subtract", "multiply", "average"], {"default": "average"}),
        })
        
        return {
            "required": base_required,
            "optional": base_inputs["optional"],
        }

    CATEGORY = "RyanOnTheInside/DepthModifiers"

    @classmethod
    def get_modifiable_params(cls):
        return ["strength", "None"]

    def apply_effect_internal(self, depth_map: np.ndarray, other_depth_maps, blend_mode, **kwargs) -> np.ndarray:
        strength = kwargs.get('strength', 1.0)
        frame_index = kwargs.get('frame_index')
        
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
    

@apply_tooltips
class DepthRippleEffect(FlexDepthBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs["required"]
        
        # Update feature_param first
        base_required["feature_param"] = cls.get_modifiable_params()
        

        # Then add other inputs
        base_required.update({
            "ripple_amplitude": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
            "ripple_frequency": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 100.0, "step": 1.0}),
            "ripple_phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 6.2832, "step": 0.1}),  # 2π
            "curvature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        
        return {
            "required": base_required,
            "optional": base_inputs["optional"],
        }

    CATEGORY = "RyanOnTheInside/DepthModifiers"

    @classmethod
    def get_modifiable_params(cls):
        return ["ripple_amplitude", "ripple_frequency", "ripple_phase", "curvature", "strength", "None"]

    def apply_effect_internal(self, depth_map: np.ndarray, frame_index, **kwargs) -> np.ndarray:
        ripple_amplitude = kwargs.get('ripple_amplitude')
        ripple_frequency = kwargs.get('ripple_frequency')
        ripple_phase = kwargs.get('ripple_phase')
        curvature = kwargs.get('curvature')
        strength = kwargs.get('strength', 1.0)

        h, w, c = depth_map.shape

        # Create a coordinate grid normalized to [0,1]
        y, x = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')

        # Center coordinates around 0
        x_c = x - 0.5
        y_c = y - 0.5

        # Compute distance from center for circular ripples
        distance = np.sqrt(x_c**2 + y_c**2)

        # Compute linear ripple pattern (lines)
        linear_pattern = x_c + y_c

        # Interpolate between linear and circular patterns
        pattern = (1 - curvature) * linear_pattern + curvature * distance

        # Compute ripple effect
        ripple = ripple_amplitude * np.sin(2 * np.pi * ripple_frequency * pattern + ripple_phase)

        # Apply ripple to depth map
        ripple = np.expand_dims(ripple, axis=-1).repeat(c, axis=-1)
        modified_depth = depth_map + ripple

        # Blend with the original depth map based on strength
        modified_depth = depth_map * (1 - strength) + modified_depth * strength

        # Ensure depth values are within valid range
        modified_depth = np.clip(modified_depth, 0.0, 1.0)

        return modified_depth


