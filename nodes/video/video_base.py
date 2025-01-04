import torch
import numpy as np
from abc import ABC, abstractmethod
from ..flex.flex_base import FlexBase
from comfy.utils import ProgressBar
from ...tooltips import apply_tooltips

@apply_tooltips
class FlexVideoBase(FlexBase, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "images": ("IMAGE",),
            }
        }

    CATEGORY = "RyanOnTheInside/FlexBase"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"

    def apply_effect(self, images, strength, feature_mode, feature_threshold, opt_feature=None, **kwargs):
        if opt_feature is None:
            return (images,)

        images_np = images.cpu().numpy()  # Convert tensor to numpy array
        num_frames = images_np.shape[0]

        # Get feature values for each frame
        feature_values = np.array([opt_feature.get_value_at_frame(i) for i in range(num_frames)])
        
        # Apply threshold to feature values
        feature_values[feature_values < feature_threshold] = 0

        # Modulate parameters based on the feature values
        for param_name in self.get_modifiable_params():
            if param_name in kwargs:
                param_value = kwargs[param_name]
                avg_feature_value = np.mean(feature_values)
                kwargs[param_name] = self.modulate_param(param_name, param_value, avg_feature_value, strength, feature_mode)

        # Apply the effect to the entire video
        processed_video = self.apply_effect_internal(images_np, feature_values=feature_values, **kwargs)

        # Convert the numpy array back to a tensor and ensure it's in BHWC format
        result_tensor = torch.from_numpy(processed_video).float()
        if result_tensor.shape[1] == 3:  # If in BCHW format, convert to BHWC
            result_tensor = result_tensor.permute(0, 2, 3, 1)

        return (result_tensor,)

    @abstractmethod
    def apply_effect_internal(self, video: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect to the entire video. To be implemented by child classes."""
        pass