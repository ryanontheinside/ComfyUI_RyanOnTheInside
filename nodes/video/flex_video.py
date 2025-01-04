import numpy as np
import torch
from .video_base import FlexVideoBase
from scipy.interpolate import interp1d
from ..masks.mask_utils import calculate_optical_flow
import cv2
import comfy.model_management as mm
from ...tooltips import apply_tooltips

from .flex_video_speed import FlexVideoSpeed

@apply_tooltips
class FlexVideoDirection(FlexVideoBase):

    @classmethod
    def get_modifiable_params(cls):
        return ["direction"]

    def apply_effect_internal(
        self,
        video: np.ndarray,
        feature_values: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        num_frames = video.shape[0]
        
        # Normalize feature values to the range [0, 1]
        normalized_features = np.clip(feature_values, 0.0, 1.0)

        # Map feature values to frame indices in the input video
        frame_indices = (normalized_features * (num_frames - 1)).astype(int)

        # Ensure frame indices stay within valid bounds
        frame_indices = np.clip(frame_indices, 0, num_frames - 1)

        # Create the processed video by selecting frames based on the mapped indices
        processed_video = video[frame_indices]

        return processed_video

@apply_tooltips
class FlexVideoSeek(FlexVideoBase):

    @classmethod
    def get_modifiable_params(cls):
        return ["seek"]

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()

        # Override the feature_mode input to remove the "absolute" option
        inputs["required"]["feature_mode"] = (["relative"], {"default": "relative"})
        # Add the new reverse parameter
        inputs["required"]["reverse"] = ("BOOLEAN", {"default": False})
        return inputs

    FUNCTION = "apply_effect"

    def apply_effect_internal(
        self,
        video: np.ndarray,
        feature_values: np.ndarray,
        reverse: bool,
        **kwargs,
    ) -> np.ndarray:
        num_frames = video.shape[0]
        processed_video = np.empty_like(video)
        strength = kwargs.get('strength', 1.0)
        seek_speed = 1.0

        # Reverse the video if the reverse parameter is True
        if reverse:
            video = video[::-1]

        # Clip feature values between 0 and 1
        feature_values_clipped = np.clip(feature_values, 0.0, 1.0)

        # Apply seek_speed to feature values
        adjusted_speeds = feature_values_clipped * seek_speed * strength

        # Ensure the total adjusted speed matches the number of frames
        total_speed = np.sum(adjusted_speeds)
        if total_speed == 0:
            adjusted_speeds = np.ones(num_frames)
        else:
            adjusted_speeds = adjusted_speeds / total_speed * num_frames

        # Calculate cumulative frame positions
        cumulative_positions = np.cumsum(adjusted_speeds)

        # Map frame indices based on cumulative positions
        frame_indices = np.clip(cumulative_positions.astype(int), 0, num_frames - 1)

        # Create the processed video by selecting frames based on the mapped indices
        for idx in range(num_frames):
            processed_video[idx] = video[frame_indices[idx]]

        return processed_video


@apply_tooltips
class FlexVideoFrameBlend(FlexVideoBase):

    @classmethod
    def get_modifiable_params(cls):
        return ["blend_strength", "frame_offset_ratio", "direction_bias", "blend_mode", "motion_blur_strength"]

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "blend_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "frame_offset_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            "direction_bias": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "blend_mode": (["normal", "additive", "multiply", "screen"],),
            "motion_blur_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        })
        return inputs

    FUNCTION = "apply_effect"

    def apply_effect_internal(
        self,
        video: np.ndarray,
        feature_values: np.ndarray,
        blend_strength: float,
        frame_offset_ratio: float,
        direction_bias: float,
        blend_mode: str,
        motion_blur_strength: float,
        **kwargs,
    ) -> np.ndarray:
        num_frames = video.shape[0]
        strength = kwargs.get('strength', 1.0)
        feature_mode = kwargs.get('feature_mode', 'relative')
        processed_video = np.empty_like(video)

        # Adjust parameters based on feature values
        if feature_mode == "relative":
            adjusted_blend = blend_strength + (feature_values - 0.5) * 2 * strength
            adjusted_offset_ratio = frame_offset_ratio + (feature_values - 0.5) * 2 * strength
            adjusted_direction_bias = direction_bias + (feature_values - 0.5) * 2 * strength
            adjusted_motion_blur = motion_blur_strength + (feature_values - 0.5) * 2 * strength
        else:  # "absolute"
            adjusted_blend = feature_values * strength
            adjusted_offset_ratio = frame_offset_ratio * feature_values * strength
            adjusted_direction_bias = direction_bias * feature_values * strength
            adjusted_motion_blur = motion_blur_strength * feature_values * strength

        adjusted_blend = np.clip(adjusted_blend, 0.0, 1.0)
        adjusted_offset_ratio = np.clip(adjusted_offset_ratio, 0.0, 1.0)
        adjusted_direction_bias = np.clip(adjusted_direction_bias, 0.0, 1.0)
        adjusted_motion_blur = np.clip(adjusted_motion_blur, 0.0, 1.0)

        max_offset = num_frames // 2
        frame_offsets = (adjusted_offset_ratio * max_offset).astype(int)

        self.start_progress(num_frames)
        for idx in range(num_frames):
            current_frame = video[idx]
            
            forward_idx = min(idx + frame_offsets[idx], num_frames - 1)
            backward_idx = max(idx - frame_offsets[idx], 0)
            
            forward_frame = video[forward_idx]
            backward_frame = video[backward_idx]

            # Blend based on direction bias
            blend_frame = adjusted_direction_bias[idx] * forward_frame + (1 - adjusted_direction_bias[idx]) * backward_frame
            
            # Apply blend mode
            if blend_mode == "additive":
                blended_frame = np.clip(current_frame + blend_frame * adjusted_blend[idx], 0.0, 1.0)
            elif blend_mode == "multiply":
                blended_frame = current_frame * (1 + (blend_frame - 0.5) * 2 * adjusted_blend[idx])
            elif blend_mode == "screen":
                blended_frame = 1 - (1 - current_frame) * (1 - blend_frame * adjusted_blend[idx])
            else:  # normal blend
                blended_frame = (1 - adjusted_blend[idx]) * current_frame + adjusted_blend[idx] * blend_frame

            # Apply motion blur
            if adjusted_motion_blur[idx] > 0:
                blur_strength = int(adjusted_motion_blur[idx] * 10) * 2 + 1  # Ensure odd number
                blended_frame = cv2.GaussianBlur(blended_frame, (blur_strength, blur_strength), 0)

            processed_video[idx] = np.clip(blended_frame, 0.0, 1.0)
            
            self.update_progress()
        self.end_progress() 

        return processed_video