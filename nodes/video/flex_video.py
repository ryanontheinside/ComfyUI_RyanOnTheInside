import numpy as np
import torch
from .video_base import FlexVideoBase
from ..flex.feature_pipe import FeaturePipe
from scipy.interpolate import interp1d
from ..masks.mask_utils import calculate_optical_flow
import cv2
import comfy.model_management as mm

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



import os
import torch
from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir
import folder_paths
import traceback
import torch
from vfi_utils import load_file_from_github_release, preprocess_frames, postprocess_frames
from packaging import version

# Add these constants at the top of the file
BASE_MODEL_DOWNLOAD_URLS = [
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/",
    "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/"
]

RIFE_CKPT_NAME_VER_DICT = {
    "rife47.pth": "4.7",
    "rife49.pth": "4.7",
}

class FlexVideoSpeed(FlexVideoBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "feature_pipe": ("FEATURE_PIPE",),
                "speed_factor": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.1}),
                "interpolation_mode": (["none", "linear", "Farneback", "rife47", "rife49"],),
                "fast_mode": ("BOOLEAN", {"default": True}),
                "ensemble": ("BOOLEAN", {"default": True}),
                "scale_factor": ([0.25, 0.5, 1.0, 2.0, 4.0], {"default": 1.0}),
            }
        }
    
    @classmethod
    def get_modifiable_params(cls):
        return ["speed_factor"]

    def apply_effect_internal(self, video: np.ndarray, feature_values: np.ndarray, feature_pipe: FeaturePipe, 
                              speed_factor: float, interpolation_mode: str, fast_mode: bool, ensemble: bool, 
                              scale_factor: float, **kwargs):
        num_frames = video.shape[0]
        total_duration = feature_pipe.frame_count / feature_pipe.frame_rate
        frame_rate = feature_pipe.frame_rate

        # Ensure feature_values is the same length as video frames
        if len(feature_values) != num_frames:
            raise ValueError("feature_values length must match the number of video frames")

        # Adjust feature values based on speed_factor
        if speed_factor >= 0:
            adjusted_feature_values = 1 - feature_values
        else:
            adjusted_feature_values = feature_values

        # Calculate frame durations based on adjusted feature values and speed factor
        base_duration = 1 / frame_rate
        adjusted_frame_durations = base_duration + (adjusted_feature_values * abs(speed_factor) * base_duration)

        cumulative_times = np.cumsum(adjusted_frame_durations)

        # Normalize cumulative times to match the total desired duration
        normalized_cumulative_times = cumulative_times * (total_duration / cumulative_times[-1])

        # Create an array of the target timestamps that the final video must have
        target_timestamps = np.linspace(0, total_duration, feature_pipe.frame_count)

        # Interpolate the adjusted frames based on the normalized timestamps
        frame_indices = np.interp(target_timestamps, normalized_cumulative_times, np.arange(num_frames))
        
        # Perform frame interpolation based on the selected mode
        if interpolation_mode == "linear":
            adjusted_video = self.linear_interpolation(video, frame_indices)
        elif interpolation_mode in ["rife47", "rife49"]:
            adjusted_video = self.rife_interpolation(video, frame_indices, interpolation_mode, fast_mode, ensemble, scale_factor)
        elif interpolation_mode in ["Farneback", "LucasKanade", "PyramidalLK"]:
            adjusted_video = self.calculate_optical_flow(video, frame_indices, interpolation_mode)
        else:  # "none"
            adjusted_video = self.no_interpolation(video, frame_indices)

        return adjusted_video

    def no_interpolation(self, video, frame_indices):
        return video[np.round(frame_indices).astype(int)]

    def linear_interpolation(self, video, frame_indices):
        num_frames, height, width, channels = video.shape
        
        # Create interpolation functions for each channel
        interp_funcs = []
        self.start_progress(channels)
        for c in range(channels):
            interp_func = interp1d(np.arange(num_frames), video[:, :, :, c], axis=0, kind='linear')
            interp_funcs.append(interp_func)
            self.update_progress()
        self.end_progress()
        
        # Apply interpolation
        interpolated_frames = np.stack([func(frame_indices) for func in interp_funcs], axis=-1)
        
        return np.clip(interpolated_frames, 0, 1)

    def calculate_optical_flow(self, video, frame_indices, interpolation_mode):
        num_frames, height, width, channels = video.shape
        interpolated_frames = []

        self.start_progress(len(frame_indices) - 1)
        for i in range(len(frame_indices) - 1):
            idx1, idx2 = int(frame_indices[i]), int(frame_indices[i + 1])
            frame1, frame2 = video[idx1], video[idx2]
            
            if idx1 == idx2:
                interpolated_frames.append(frame1)
            else:
                # Convert frames to 8-bit unsigned integer format


                flow = calculate_optical_flow(frame1, frame2, interpolation_mode)
                
                # Calculate the fractional part for interpolation
                frac = frame_indices[i] - idx1
                
                # Perform the interpolation
                h, w = flow.shape[:2]
                flow_x, flow_y = flow[:,:,0], flow[:,:,1]
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                
                interp_x = np.clip(x + frac * flow_x, 0, w - 1).astype(np.float32)
                interp_y = np.clip(y + frac * flow_y, 0, h - 1).astype(np.float32)
                
                interpolated_frame = cv2.remap(frame1, interp_x, interp_y, cv2.INTER_LINEAR)
                interpolated_frames.append(interpolated_frame)
            
            self.update_progress()
        
        self.end_progress()
        
        return np.array(interpolated_frames)

    def rife_interpolation(self, video, frame_indices, interpolation_mode, fast_mode, ensemble, scale_factor):
        from .rife_arch import IFNet
        import comfy.model_management as mm

        ckpt_name = f"{interpolation_mode}.pth"
        model_path = self.load_file_from_github_release("rife", ckpt_name)
        arch_ver = RIFE_CKPT_NAME_VER_DICT[ckpt_name]
        
        device = mm.get_torch_device()
        interpolation_model = IFNet(arch_ver=arch_ver)
        interpolation_model.load_state_dict(torch.load(model_path))
        interpolation_model.eval().to(device)
        
        # Input video is already in BHWC format, so we don't need to permute
        video_tensor = torch.from_numpy(video).float() / 255.0
        video_tensor = preprocess_frames(video_tensor).to(device)
        
        scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]
        
        interpolated_frames = []
        self.start_progress(len(frame_indices))
        
        for i in range(len(frame_indices)):
            current_idx = frame_indices[i]
            lower_idx = int(np.floor(current_idx))
            upper_idx = int(np.ceil(current_idx))
            
            if lower_idx == upper_idx:
                interpolated_frames.append(video_tensor[lower_idx:lower_idx+1])
            else:
                frame1 = video_tensor[lower_idx:lower_idx+1]
                frame2 = video_tensor[upper_idx:upper_idx+1]
                timestep = current_idx - lower_idx
                middle_frame = interpolation_model(frame1, frame2, timestep, scale_list, fast_mode, ensemble)
                interpolated_frames.append(middle_frame)
            
            self.update_progress()
        
        self.end_progress()
        
        interpolated_video = torch.cat(interpolated_frames, dim=0)
        # Ensure output is in BHWC format and convert back to numpy array
        return (postprocess_frames(interpolated_video).cpu() * 255.0).numpy()

    @staticmethod
    def get_ckpt_container_path(model_type):
        # Use the original save location
        return os.path.join(folder_paths.models_dir, model_type)

    @staticmethod
    def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
        if model_dir is None:
            hub_dir = get_dir()
            model_dir = os.path.join(hub_dir, 'checkpoints')

        os.makedirs(model_dir, exist_ok=True)

        parts = urlparse(url)
        file_name = os.path.basename(parts.path) if file_name is None else file_name
        cached_file = os.path.abspath(os.path.join(model_dir, file_name))
        if not os.path.exists(cached_file):
            print(f'Downloading: "{url}" to {cached_file}\n')
            download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
        return cached_file

    @classmethod
    def load_file_from_github_release(cls, model_type, ckpt_name):
        error_strs = []
        for i, base_model_download_url in enumerate(BASE_MODEL_DOWNLOAD_URLS):
            try:
                return cls.load_file_from_url(base_model_download_url + ckpt_name, cls.get_ckpt_container_path(model_type))
            except Exception:
                traceback_str = traceback.format_exc()
                if i < len(BASE_MODEL_DOWNLOAD_URLS) - 1:
                    print("Failed! Trying another endpoint.")
                error_strs.append(f"Error when downloading from: {base_model_download_url + ckpt_name}\n\n{traceback_str}")

        error_str = '\n\n'.join(error_strs)
        raise Exception(f"Tried all GitHub base urls to download {ckpt_name} but no success. Below is the error log:\n\n{error_str}")

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