import os
import torch
from urllib.parse import urlparse
from torch.hub import download_url_to_file, get_dir
import folder_paths
import traceback
import torch
from .vfi_utils import preprocess_frames, postprocess_frames
from .video_base import FlexVideoBase
import numpy as np
from scipy.interpolate import interp1d
from ..masks.mask_utils import calculate_optical_flow
import cv2
import comfy.model_management as mm
from ...tooltips import apply_tooltips

BASE_MODEL_DOWNLOAD_URLS = [
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/",
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/download/models/",
    "https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.0/"
]

RIFE_CKPT_NAME_VER_DICT = {
    "rife47.pth": "4.7",
    "rife49.pth": "4.7",
}

@apply_tooltips
class FlexVideoSpeed(FlexVideoBase):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()
        parent_inputs["required"]["feature_param"] = cls.get_modifiable_params()
        return {
            **parent_inputs,  # Keep all parent inputs including optional
            "required": {

                **parent_inputs["required"],
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

    def apply_effect_internal(self, video: np.ndarray, feature_values: np.ndarray, 
                              speed_factor: np.ndarray, interpolation_mode: str, fast_mode: bool, ensemble: bool, 
                              scale_factor: float, opt_feature=None, **kwargs):
        num_frames = video.shape[0]
        frame_rate = opt_feature.frame_rate
        total_duration = num_frames / frame_rate

        # Ensure feature_values is the same length as video frames
        if len(feature_values) != num_frames:
            raise ValueError("feature_values length must match the number of video frames")

        # Get threshold from kwargs
        feature_threshold = kwargs.get('feature_threshold', 0.0)

        # Apply threshold - zero out values below threshold
        above_threshold = feature_values >= feature_threshold
        feature_values = np.where(above_threshold, feature_values, 0.0)

        # Adjust feature values based on speed_factor (element-wise comparison)
        adjusted_feature_values = np.where(speed_factor >= 0, 1 - feature_values, feature_values)

        # Calculate frame durations based on adjusted feature values and speed factor
        base_duration = 1 / frame_rate
        adjusted_frame_durations = base_duration + (adjusted_feature_values * np.abs(speed_factor) * base_duration)

        cumulative_times = np.cumsum(adjusted_frame_durations)

        # Normalize cumulative times to match the total desired duration
        normalized_cumulative_times = cumulative_times * (total_duration / cumulative_times[-1])

        # Create an array of the target timestamps that the final video must have
        target_timestamps = np.linspace(0, total_duration, num_frames)

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
