import numpy as np
import torch
import cv2
from .mask_base import FlexMaskBase

class FlexMaskNormalBase(FlexMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "normal_map": ("IMAGE",),
            }
        }

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def normal_map_to_array(self, normal_map: torch.Tensor) -> np.ndarray:
        # Convert normal map from [0, 1] to [-1, 1] range
        normal_array = normal_map.cpu().numpy() * 2 - 1
        return normal_array

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, normal_map, **kwargs):
        # This method should be overridden by child classes
        raise NotImplementedError("Subclasses must implement main_function")

class FlexMaskNormalLighting(FlexMaskNormalBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "light_direction_x": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01}),
                "light_direction_y": ("FLOAT", {"default": -0.5, "min": -1.0, "max": 1.0, "step": 0.01}),
                "light_direction_z": ("FLOAT", {"default": 0.7, "min": -1.0, "max": 1.0, "step": 0.01}),
                "shadow_threshold": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_param": (["none", "direction", "threshold", "both"],),
                "feature_mode": (["rotate", "intensity"],),
            }
        }

    CATEGORY = "RyanOnTheInside/FlexMasks"

    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, 
                     normal_map: torch.Tensor, light_direction_x: float, light_direction_y: float, 
                     light_direction_z: float, shadow_threshold: float, feature_param: str, 
                     feature_mode: str, **kwargs) -> np.ndarray:
        
        normal_array = self.normal_map_to_array(normal_map)
        
        # Normalize light direction
        light_direction = self.normalize_vector(np.array([light_direction_x, light_direction_y, light_direction_z]))
        
        # Apply feature modulation
        if feature_param != "none":
            if feature_mode == "rotate":
                # Rotate light direction based on feature value
                rotation_angle = 2 * np.pi * feature_value * strength
                rotation_matrix = np.array([
                    [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                    [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                    [0, 0, 1]
                ])
                light_direction = np.dot(rotation_matrix, light_direction)
            elif feature_mode == "intensity":
                # Modify shadow threshold based on feature value
                shadow_threshold += (feature_value - 0.5) * strength
                shadow_threshold = np.clip(shadow_threshold, 0.0, 1.0)
        
        # Calculate dot product between normal vectors and light direction
        dot_product = np.einsum('bhwc,c->bhw', normal_array, light_direction)
        
        # Create lighting mask
        lighting_mask = (dot_product > shadow_threshold).astype(np.float32)
        
        # Combine with input mask
        combined_mask = mask * lighting_mask
        
        return combined_mask

    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, normal_map, light_direction_x, light_direction_y, light_direction_z, shadow_threshold, feature_param, feature_mode, **kwargs):
        processed_masks = []
        for i in range(masks.shape[0]):  # Iterate over all frames
            mask = masks[i].cpu().numpy()
            feature_value = feature.get_value_at_frame(i)
            
            if feature_value >= feature_threshold:
                processed_mask = self.process_mask(
                    mask, feature_value, strength, normal_map[i], 
                    light_direction_x, light_direction_y, light_direction_z, 
                    shadow_threshold, feature_param, feature_mode
                )
            else:
                processed_mask = mask
            
            processed_masks.append(processed_mask)
        
        processed_masks = np.stack(processed_masks)
        return self.apply_mask_operation(torch.from_numpy(processed_masks), masks, strength, invert, subtract_original, grow_with_blur)