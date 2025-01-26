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
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()

        # Override the feature_mode input to remove the "absolute" option
        inputs["required"]["feature_mode"] = (["relative"], {"default": "relative"})
        inputs["required"]["feature_param"] = cls.get_modifiable_params()
        
        return inputs
        
    @classmethod
    def get_modifiable_params(cls):
        return ["direction"]

    def apply_effect_internal(
        self,
        video: np.ndarray,
        feature_values: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        num_output_frames = len(feature_values)
        num_input_frames = video.shape[0]
        
        # Ensure strength and threshold arrays match feature length
        strength = kwargs.get('strength')
        feature_threshold = kwargs.get('feature_threshold')
        
        # If arrays are wrong length (because they were based on video length), resize them
        if len(strength) != num_output_frames:
            if len(strength) == 1:  # If it's a single value expanded
                strength = np.full(num_output_frames, strength[0])
            else:  # If it's a different length array
                strength = np.broadcast_to(strength, (num_output_frames,))
            
        if len(feature_threshold) != num_output_frames:
            if len(feature_threshold) == 1:
                feature_threshold = np.full(num_output_frames, feature_threshold[0])
            else:
                feature_threshold = np.broadcast_to(feature_threshold, (num_output_frames,))
        
        # Normalize feature values to 0-1
        feature_min = np.min(feature_values)
        feature_max = np.max(feature_values)
        if feature_max > feature_min:
            normalized_features = (feature_values - feature_min) / (feature_max - feature_min)
            normalized_threshold = (feature_threshold - feature_min) / (feature_max - feature_min)
        else:
            normalized_features = np.full_like(feature_values, 0.5)
            normalized_threshold = feature_threshold

        # Create output video array matching feature length
        processed_video = np.empty((num_output_frames, *video.shape[1:]), dtype=video.dtype)
        
        # Handle thresholding
        above_threshold = normalized_features >= normalized_threshold
        
        # For frames below threshold, use first frame
        frame_positions = np.where(
            above_threshold,
            normalized_features * (num_input_frames - 1) * strength,
            np.zeros_like(normalized_features)
        )
        
        # Convert to integer indices with clipping
        frame_indices = np.clip(frame_positions.astype(int), 0, num_input_frames - 1)
        
        # Create the output video by sampling from input video
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
        inputs["required"]["feature_param"] = cls.get_modifiable_params()
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
        strength = kwargs.get('strength', np.ones(num_frames))
        feature_threshold = kwargs.get('feature_threshold', np.zeros(num_frames))
        seek_speed = 1.0

        # Reverse the video if the reverse parameter is True
        if reverse:
            video = video[::-1]

        # Create a mask for values above threshold (element-wise comparison)
        above_threshold = feature_values >= feature_threshold
        
        # For values below threshold, we'll use 0 speed (stay on current frame)
        # For values above threshold, we'll use the normalized feature value
        feature_values_masked = np.where(above_threshold, 
                                       np.clip(feature_values, 0.0, 1.0), 
                                       0.0)

        # Apply seek_speed and strength (element-wise multiplication)
        adjusted_speeds = feature_values_masked * seek_speed * strength

        # If all speeds are 0 (all below threshold), use uniform movement
        total_speed = np.sum(adjusted_speeds)
        if total_speed == 0:
            adjusted_speeds = np.ones(num_frames)
        else:
            adjusted_speeds = adjusted_speeds / total_speed * num_frames

        # Calculate cumulative positions
        cumulative_positions = np.cumsum(adjusted_speeds)
        
        # For frames where feature is below threshold, maintain previous frame
        frame_indices = np.zeros(num_frames, dtype=int)
        last_valid_idx = 0
        
        for i in range(num_frames):
            if above_threshold[i]:
                frame_idx = int(np.clip(cumulative_positions[i], 0, num_frames - 1))
                last_valid_idx = frame_idx
            else:
                frame_idx = last_valid_idx
            frame_indices[i] = frame_idx

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