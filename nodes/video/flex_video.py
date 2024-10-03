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

class FlexVideoSeek(FlexVideoBase):

    @classmethod
    def get_modifiable_params(cls):
        return ["seek_speed"]

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "seek_speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
        })
        return inputs

    FUNCTION = "apply_effect"

    def apply_effect_internal(
        self,
        video: np.ndarray,
        feature_values: np.ndarray,
        seek_speed: float,
        **kwargs,
    ) -> np.ndarray:
        num_frames = video.shape[0]
        processed_video = np.empty_like(video)
        # Get strength from kwargs
        strength = kwargs.get('strength', 1.0)
        # Compute cumulative feature values to determine frame positions
        feature_values_clipped = np.clip(feature_values, 0.0, 1.0)  # Ensure values are between 0 and 1
        adjusted_speeds = feature_values_clipped * seek_speed  * strength# Adjust speeds according to feature values

        # Normalize adjusted speeds to ensure total frames match input
        total_speed = np.sum(adjusted_speeds)
        if total_speed == 0:
            adjusted_speeds = np.ones(num_frames) * (num_frames / num_frames)
        else:
            adjusted_speeds = adjusted_speeds / total_speed * (num_frames - 1)

        cumulative_speeds = np.cumsum(adjusted_speeds)

        # Map frame indices based on cumulative speeds
        frame_indices = np.clip(cumulative_speeds.astype(int), 0, num_frames - 1)

        # Ensure the output video has the same number of frames
        for idx in range(num_frames):
            frame_idx = frame_indices[idx]
            processed_video[idx] = video[frame_idx]

        return processed_video


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