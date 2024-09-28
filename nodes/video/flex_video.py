import numpy as np
from .video_base import FlexVideoBase

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

class FlexVideoSpeed(FlexVideoBase):

    @classmethod
    def get_modifiable_params(cls):
        return ["max_speed_percent"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "feature_pipe": ("FEATURE_PIPE",),
                "max_speed_percent": ("FLOAT", {"default": 500.0, "min": 1.0, "max": 1000.0, "step": 1.0}),
                "duration_adjustment_method": (["Interpolate", "Truncate/Repeat"],),
            }
        }

    FUNCTION = "apply_effect"

    def apply_effect_internal(
        self,
        video: np.ndarray,
        feature_values: np.ndarray,
        max_speed_percent: float,
        feature_pipe=None,
        duration_adjustment_method="Interpolate",
        **kwargs,
    ) -> np.ndarray:
        strength = kwargs.get('strength', 1.0)
        feature_mode = kwargs.get('feature_mode', 'relative')
        num_frames = video.shape[0]
        original_frame_rate = feature_pipe.frame_rate
        num_features = len(feature_values)

        # Adjust video duration to match the number of features
        if num_frames != num_features:
            if duration_adjustment_method == "Interpolate":
                video = self.interpolate_video(video, num_features)
            else:  # "Truncate/Repeat"
                video = self.truncate_or_repeat_video(video, num_features)
            num_frames = num_features

        # Convert max_speed_percent to a speed factor
        max_speed_factor = max_speed_percent / 100.0

        # Apply strength and feature_mode to adjust speed
        if feature_mode == "relative":
            # Feature value of 0.5 means original speed
            speed_factors = 1.0 + (feature_values - 0.5) * 2 * (max_speed_factor - 1.0) * strength
        else:  # "absolute"
            # Feature values directly map to speed, scaled by max_speed_factor
            speed_factors = feature_values * max_speed_factor * strength

        # Ensure speed factors are within a reasonable range
        speed_factors = np.clip(speed_factors, 1e-6, max_speed_factor)

        # Compute time intervals between frames based on adjusted speed
        time_intervals = 1.0 / (original_frame_rate * speed_factors)

        # Compute cumulative time
        cumulative_time = np.cumsum(time_intervals)
        total_time = cumulative_time[-1]

        # Generate new time stamps for the output frames
        output_time_stamps = np.linspace(0, total_time, num_frames)

        # Map output time stamps to input frame indices via interpolation
        new_frame_indices = np.interp(output_time_stamps, cumulative_time, np.arange(num_frames))

        # Interpolate frames
        processed_video = np.empty_like(video)
        self.start_progress(len(new_frame_indices))
        for idx, t in enumerate(new_frame_indices):
            lower_idx = int(np.floor(t))
            upper_idx = min(lower_idx + 1, num_frames - 1)
            weight = t - lower_idx
            processed_video[idx] = (1 - weight) * video[lower_idx] + weight * video[upper_idx]
            self.update_progress()
        self.end_progress()
        return processed_video
    
    def interpolate_video(self, video: np.ndarray, target_frames: int) -> np.ndarray:
        """Interpolate video to match the target number of frames."""
        orig_frames = video.shape[0]
        if orig_frames == target_frames:
            return video
        
        new_video = np.empty((target_frames, *video.shape[1:]), dtype=video.dtype)
        for i in range(target_frames):
            t = i / (target_frames - 1) * (orig_frames - 1)
            idx1, idx2 = int(t), min(int(t) + 1, orig_frames - 1)
            weight = t - idx1
            new_video[i] = (1 - weight) * video[idx1] + weight * video[idx2]
        
        return new_video

    def truncate_or_repeat_video(self, video: np.ndarray, target_frames: int) -> np.ndarray:
        """Truncate or repeat video frames to match the target number of frames."""
        orig_frames = video.shape[0]
        if orig_frames == target_frames:
            return video
        
        if orig_frames > target_frames:
            return video[:target_frames]
        else:
            repeats = target_frames // orig_frames
            remainder = target_frames % orig_frames
            return np.concatenate([np.tile(video, (repeats, 1, 1, 1)), video[:remainder]])
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