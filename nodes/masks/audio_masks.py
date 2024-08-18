from .mask_base import MaskBase
from .mask_utils import morph_mask, normalize_array, warp_mask, transform_mask, combine_masks
from ..audio.audio_utils import AudioFeatureExtractor
import numpy as np
import torch

class AudioMaskBase(MaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "audio": ("AUDIO",),
                "video_frames": ("IMAGE",),
                "audio_feature": (["amplitude_envelope", "rms_energy", "spectral_centroid", "onset_detection", "chroma_features"],),
                "feature_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frame_rate": ("FLOAT", {"default": 30, "min": 0.1, "max": 120, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MASK","AUDIO")
    FUNCTION = "main_function"

    def __init__(self):
        super().__init__()

    def initialize(self, audio, audio_feature, num_frames, frame_rate):
        self.audio_feature_extractor = AudioFeatureExtractor(audio, num_frames, frame_rate, feature_type=audio_feature)

    def process_mask(self, mask: np.ndarray, strength: float, audio: dict, video_frames: torch.Tensor, audio_feature: str, feature_threshold: float, frame_rate: float, **kwargs) -> np.ndarray:
        # Extract audio feature
        feature = self.audio_feature_extractor.extract()
        
        # Normalize feature to [0, 1] range
        normalized_feature = normalize_array(feature)
        
        # Ensure the feature length matches the number of video frames
        num_frames = mask.shape[0]
        if len(normalized_feature) != num_frames:
            normalized_feature = np.interp(np.linspace(0, 1, num_frames), np.linspace(0, 1, len(normalized_feature)), normalized_feature)
        
        # Apply the audio-driven effect to each frame
        processed_masks = []
        for i in range(num_frames):
            if normalized_feature[i] < feature_threshold:
                processed_masks.append(mask[i])
            else:
                frame_strength = strength * (1 + normalized_feature[i])
                processed_mask = self.process_single_frame(mask[i], frame_strength, frame_index=i, **kwargs)
                processed_masks.append(processed_mask)
        
        return np.stack(processed_masks)

    def process_single_frame(self, mask: np.ndarray, strength: float, frame_index: int = None, **kwargs) -> np.ndarray:
        """
        Process a single frame of the mask.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement process_single_frame method")

    def apply_audio_driven_mask(self, masks, audio, video_frames, strength, audio_feature, feature_threshold, frame_rate, **kwargs):
        processed_masks = self.process_mask(masks.numpy(), strength, audio, video_frames, audio_feature, feature_threshold, frame_rate, **kwargs)
        return self.apply_mask_operation(torch.from_numpy(processed_masks), masks, strength, **kwargs)

    def main_function(self, masks, audio, video_frames, strength, audio_feature, feature_threshold, frame_rate, **kwargs):
        num_frames = masks.shape[0]
        self.initialize(audio, audio_feature, num_frames, frame_rate)
        
        return self.apply_audio_driven_mask(masks, audio, video_frames, strength, audio_feature, feature_threshold, frame_rate, **kwargs), audio

class AudioMaskMorph(AudioMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "morph_type": (["erode", "dilate", "open", "close"],),
                "max_kernel_size": ("INT", {"default": 5, "min": 3, "max": 21, "step": 2}),
                "max_iterations": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
            }
        }

    def process_single_frame(self, mask: np.ndarray, strength: float, morph_type: str, max_kernel_size: int, max_iterations: int, **kwargs) -> np.ndarray:
        kernel_size = max(3, int(3 + (max_kernel_size - 3) * strength))
        iterations = max(1, int(max_iterations * strength))
        
        return morph_mask(mask, morph_type, kernel_size, iterations)

    def main_function(self, masks, audio, video_frames, strength, audio_feature, feature_threshold, frame_rate, morph_type, max_kernel_size, max_iterations, **kwargs):
        return super().main_function(masks, audio, video_frames, strength, audio_feature, feature_threshold, frame_rate, 
                                     morph_type=morph_type, max_kernel_size=max_kernel_size, max_iterations=max_iterations, **kwargs)

class AudioMaskWarp(AudioMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "warp_type": (["perlin", "radial", "swirl"],),
                "frequency": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
                "max_amplitude": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 500.0, "step": 0.1}),
                "octaves": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}),
            }
        }

    def process_single_frame(self, mask: np.ndarray, strength: float, warp_type: str, frequency: float, max_amplitude: float, octaves: int, **kwargs) -> np.ndarray:
        amplitude = max_amplitude * strength
        return warp_mask(mask, warp_type, frequency, amplitude, octaves)

    def main_function(self, masks, audio, video_frames, strength, audio_feature, feature_threshold, frame_rate, warp_type, frequency, max_amplitude, octaves, **kwargs):
        return super().main_function(masks, audio, video_frames, strength, audio_feature, feature_threshold, frame_rate, 
                                     warp_type=warp_type, frequency=frequency, max_amplitude=max_amplitude, octaves=octaves, **kwargs)
    
class AudioMaskTransform(AudioMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "transform_type": (["translate", "rotate", "scale"],),
                "max_x_value": ("FLOAT", {"default": 10.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "max_y_value": ("FLOAT", {"default": 10.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
            }
        }

    def process_single_frame(self, mask: np.ndarray, strength: float, transform_type: str, max_x_value: float, max_y_value: float, **kwargs) -> np.ndarray:
        x_value = max_x_value * strength
        y_value = max_y_value * strength
        return transform_mask(mask, transform_type, x_value, y_value)

    def main_function(self, masks, audio, video_frames, strength, audio_feature, feature_threshold, frame_rate, transform_type, max_x_value, max_y_value, **kwargs):
        return super().main_function(masks, audio, video_frames, strength, audio_feature, feature_threshold, frame_rate, 
                                     transform_type=transform_type, max_x_value=max_x_value, max_y_value=max_y_value, **kwargs)

class AudioMaskMath(AudioMaskBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "mask_b": ("MASK",),
                "combination_method": (["add", "subtract", "multiply", "minimum", "maximum"],),
            }
        }

    def process_single_frame(self, mask: np.ndarray, strength: float, mask_b: np.ndarray, combination_method: str, frame_index: int = None, **kwargs) -> np.ndarray:
        if frame_index is not None:
            mask_b_frame = mask_b[frame_index]
        else:
            mask_b_frame = mask_b
        return combine_masks(mask, mask_b_frame, combination_method, strength)

    def main_function(self, masks, audio, video_frames, strength, audio_feature, feature_threshold, frame_rate, mask_b, combination_method, **kwargs):
        mask_b_np = mask_b.cpu().numpy() if isinstance(mask_b, torch.Tensor) else mask_b
        return super().main_function(masks, audio, video_frames, strength, audio_feature, feature_threshold, frame_rate, 
                                     mask_b=mask_b_np, combination_method=combination_method, **kwargs)