import torch
import numpy as np
from abc import ABC, abstractmethod
from ... import RyanOnTheInside
from comfy.utils import ProgressBar

class FlexVideoBase(RyanOnTheInside, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "feature": ("FEATURE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "feature_mode": (["relative", "absolute"], {"default": "relative"}),
                "feature_param": (cls.get_modifiable_params(),),
                "feature_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "feature_pipe": ("FEATURE_PIPE",)
            }
        }

    CATEGORY = "RyanOnTheInside/VideoEffects"
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

    #TODO implement parame name
    def modulate_param(self, param_name, param_value, feature_value, strength, mode):
        if mode == "relative":
            # Adjust parameter relative to its value and the feature
            return param_value * (1 + (feature_value - 0.5) * 2 * strength)
        else:  # absolute
            # Adjust parameter directly based on the feature
            return param_value * feature_value * strength

    def apply_effect(self, images, feature, strength, feature_mode, feature_threshold, feature_pipe=None, **kwargs):
        images_np = images.cpu().numpy()  # Convert tensor to numpy array
        num_frames = images_np.shape[0]

        target_frame_count = feature_pipe.frame_count if feature_pipe is not None else num_frames
        feature_values = np.array([feature.get_value_at_frame(i) for i in range(target_frame_count)])
        
        # Apply threshold to feature values
        feature_values[feature_values < feature_threshold] = 0

        # Modulate parameters based on the feature values
        for param_name in self.get_modifiable_params():
            if param_name in kwargs:
                param_value = kwargs[param_name]
                avg_feature_value = np.mean(feature_values)
                kwargs[param_name] = self.modulate_param(param_name, param_value, avg_feature_value, strength, feature_mode)

        # Apply the effect to the entire video
        processed_video = self.apply_effect_internal(images_np, feature_values=feature_values, feature_pipe=feature_pipe, **kwargs)

        # Convert the numpy array back to a tensor and ensure it's in BHWC format
        result_tensor = torch.from_numpy(processed_video).float()
        if result_tensor.shape[1] == 3:  # If in BCHW format, convert to BHWC
            result_tensor = result_tensor.permute(0, 2, 3, 1)

        return (result_tensor,)

    @abstractmethod
    def apply_effect_internal(self, video: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect to the entire video. To be implemented by child classes."""
        pass