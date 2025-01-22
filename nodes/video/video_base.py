import torch
import numpy as np
from abc import ABC, abstractmethod
from ..flex.flex_base import FlexBase
from comfy.utils import ProgressBar
from ...tooltips import apply_tooltips

#TODO: maybe implement a batch version of flex base when the time comes. Currently, this differs significantly from its siblings
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

    CATEGORY = "RyanOnTheInside/FlexFeatures/Targets/Video"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"

    def apply_effect(self, images, strength, feature_mode, feature_threshold, feature_param, opt_feature=None, **kwargs):
        images_np = images.cpu().numpy()
        
        # Get feature length first
        if opt_feature is not None:
            num_frames = opt_feature.frame_count
        else:
            num_frames = images_np.shape[0]  # Fallback to video length if no feature

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        # Handle non-array parameters directly
        processed_kwargs = {k: v for k, v in kwargs.items() 
                          if not isinstance(v, (list, tuple, np.ndarray))}
        
        # Process parameters frame by frame and collect feature values
        feature_values = []
        for i in range(num_frames):  # Now using feature length!
            feature_value = self.get_feature_value(i, opt_feature)
            feature_value = 0.5 if feature_value is None else feature_value
            feature_values.append(feature_value)

            # Process parameters
            frame_kwargs = self.process_parameters(
                frame_index=i,
                feature_value=feature_value,
                feature_threshold=feature_threshold,
                strength=strength,
                feature_param=feature_param,
                feature_mode=feature_mode,
                **kwargs
            )
            # Store modulated values and array values
            for k, v in frame_kwargs.items():
                if k not in processed_kwargs:
                    processed_kwargs[k] = []
                if isinstance(processed_kwargs[k], list):
                    processed_kwargs[k].append(v)

        # Convert collected values to arrays
        feature_values = np.array(feature_values)
        for k, v in processed_kwargs.items():
            if isinstance(v, list):
                processed_kwargs[k] = np.array(v)

        # Convert strength and feature_threshold to arrays right before passing to child
        strength = np.asarray(strength)
        feature_threshold = np.asarray(feature_threshold)
        if strength.ndim == 0:
            strength = np.full(num_frames, strength)
        if feature_threshold.ndim == 0:
            feature_threshold = np.full(num_frames, feature_threshold)
        processed_kwargs['strength'] = strength
        processed_kwargs['feature_threshold'] = feature_threshold

        # Let child classes handle the video processing and threshold checks
        processed_video = self.apply_effect_internal(
            images_np,
            feature_values=feature_values,
            opt_feature=opt_feature,
            **processed_kwargs
        )

        self.end_progress()

        # Convert to tensor and ensure BHWC format
        result_tensor = torch.from_numpy(processed_video).float()
        if result_tensor.shape[1] == 3:  # If in BCHW format, convert to BHWC
            result_tensor = result_tensor.permute(0, 2, 3, 1)

        return (result_tensor,)

    @abstractmethod
    def apply_effect_internal(self, video: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect to the entire video. To be implemented by child classes."""
        pass