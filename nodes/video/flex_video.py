import numpy as np
import torch
from .video_base import FlexVideoBase
from ..flex.feature_pipe import FeaturePipe
from scipy.interpolate import interp1d
from ..masks.mask_utils import calculate_optical_flow
import cv2
import comfy.model_management as mm

from .flex_video_speed import FlexVideoSpeed

class FlexVideoDirection(FlexVideoBase):

    @classmethod
    def get_modifiable_params(cls):
        return ["direction"]

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        # Make feature_pipe optional
        inputs.setdefault("optional", {})
        inputs["optional"].update({
            "feature_pipe": ("FEATURE_PIPE",),
        })
        return inputs


    def apply_effect_internal(
        self,
        video: np.ndarray,
        feature_values: np.ndarray,
        feature_pipe=None,
        **kwargs,
    ) -> np.ndarray:
        num_frames = video.shape[0]
        
        # Use the frame count from the feature pipe if provided, otherwise fallback to the input video length
        if feature_pipe is not None:
            target_frame_count = feature_pipe.frame_count
        else:
            target_frame_count = num_frames

        # Normalize feature values to the range [0, 1] over the length of the feature pipe (or input video)
        normalized_features = np.clip(feature_values, 0.0, 1.0)

        # Map feature values to frame indices in the input video
        frame_indices = (normalized_features * (num_frames - 1)).astype(int)

        # Ensure frame indices stay within valid bounds
        frame_indices = np.clip(frame_indices, 0, num_frames - 1)

        # If a feature pipe is provided, the output video should be the length of the feature pipe
        if target_frame_count != num_frames:
            # Adjust the output video length to match the feature pipe length
            # Select frames based on the feature value mapping
            frame_indices = np.interp(
                np.linspace(0, 1, target_frame_count),
                np.linspace(0, 1, len(feature_values)),
                frame_indices
            ).astype(int)

        # Create the processed video by selecting frames based on the mapped indices
        processed_video = video[frame_indices]

        return processed_video




class FlexVideoFrameBlend(FlexVideoBase):

    @classmethod
    def get_modifiable_params(cls):
        return ["blend_strength"]

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return inputs

    FUNCTION = "apply_effect"

    def apply_effect_internal(
        self,
        video: np.ndarray,
        feature_values: np.ndarray,
        blend_strength: float,
        **kwargs,
    ) -> np.ndarray:
        num_frames = video.shape[0]
        strength = kwargs.get('strength', 1.0)
        feature_mode = kwargs.get('feature_mode', 'relative')
        processed_video = np.empty_like(video)

        # Adjust blend strength based on feature values
        if feature_mode == "relative":
            adjusted_blend = blend_strength + (feature_values - 0.5) * 2 * strength
        else:  # "absolute"
            adjusted_blend = feature_values * strength

        adjusted_blend = np.clip(adjusted_blend, 0.0, 1.0)

        self.start_progress(num_frames)
        for idx in range(num_frames):
            current_frame = video[idx]
            if idx < num_frames - 1:
                next_frame = video[idx + 1]
            else:
                next_frame = video[idx]  # Last frame, no next frame available

            alpha = adjusted_blend[idx]
            blended_frame = (1 - alpha) * current_frame + alpha * next_frame
            processed_video[idx] = np.clip(blended_frame, 0.0, 1.0)
            self.update_progress()
        self.end_progress() 

        return processed_video