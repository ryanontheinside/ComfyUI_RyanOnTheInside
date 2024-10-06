import torch
import numpy as np
from tqdm import tqdm
from comfy.utils import ProgressBar
from .mask_utils import (
    create_distance_transform, 
    normalize_array, 
    apply_blur, 
    apply_easing, 
    calculate_optical_flow, 
    apply_blur, 
    normalize_array
    )
from abc import ABC, abstractmethod
import pymunk 
import math
import random
from typing import List, Tuple
import pymunk
import cv2
from ..audio.audio_processor_legacy import AudioFeatureExtractor
from ... import RyanOnTheInside


class MaskBase(RyanOnTheInside, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "invert": ("BOOLEAN", {"default": False}),
                "subtract_original": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "grow_with_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            }
        }

    CATEGORY = "RyanOnTheInside/Masks"

    
    def __init__(self):
        self.pre_processors = []
        self.post_processors = []
        self.progress_bar = None
        self.tqdm_bar = None
        self.current_progress = 0
        self.total_steps = 0

    def add_pre_processor(self, func):
        self.pre_processors.append(func)
        return self

    def add_post_processor(self, func):
        self.post_processors.append(func)
        return self

    def pre_process(self, mask):
        for processor in self.pre_processors:
            mask = processor(mask)
        return mask

    def post_process(self, mask):
        for processor in self.post_processors:
            mask = processor(mask)
        return mask

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)
        self.tqdm_bar = tqdm(total=total_steps, desc=desc, leave=False)
        self.current_progress = 0
        self.total_steps = total_steps

    def update_progress(self, step=1):
        self.current_progress += step
        if self.progress_bar:
            self.progress_bar.update(step)
        if self.tqdm_bar:
            self.tqdm_bar.update(step)

    def end_progress(self):
        if self.tqdm_bar:
            self.tqdm_bar.close()
        self.progress_bar = None
        self.tqdm_bar = None
        self.current_progress = 0
        self.total_steps = 0

    @abstractmethod
    def process_mask(self, mask: np.ndarray, strength: float, **kwargs) -> np.ndarray:
        """
        Process a single mask. This method must be implemented by child classes.
        """
        pass

    def apply_mask_operation(self, processed_masks: torch.Tensor, original_masks: torch.Tensor, strength: float, invert: bool, subtract_original: float, grow_with_blur: float, **kwargs) -> Tuple[torch.Tensor]:
        processed_masks_np = processed_masks.cpu().numpy() if isinstance(processed_masks, torch.Tensor) else processed_masks
        original_masks_np = original_masks.cpu().numpy() if isinstance(original_masks, torch.Tensor) else original_masks
        num_frames = processed_masks_np.shape[0]

        self.start_progress(num_frames, desc="Applying mask operation")

        result = []
        for processed_mask, original_mask in zip(processed_masks_np, original_masks_np):
            # Pre-processing
            processed_mask = self.pre_process(processed_mask)

            if invert:
                processed_mask = 1 - processed_mask

            if grow_with_blur > 0:
                processed_mask = apply_blur(processed_mask, grow_with_blur)

            # Apply subtract_original as the final step
            if subtract_original > 0:
                dist_transform = create_distance_transform(original_mask)
                dist_transform = normalize_array(dist_transform)
                threshold = 1 - subtract_original
                subtraction_mask = dist_transform > threshold
                processed_mask[subtraction_mask] = 0

            # Post-processing
            processed_mask = self.post_process(processed_mask)

            # Ensure the final mask is clipped between 0 and 1
            processed_mask = np.clip(processed_mask, 0, 1)

            result.append(processed_mask)
            self.update_progress()

        self.end_progress()

        return torch.from_numpy(np.stack(result)).float()

    @abstractmethod
    def main_function(self, *args, **kwargs) -> Tuple[torch.Tensor]:
        """
        Main entry point for the node. This method must be implemented by child classes.
        It should call apply_mask_operation with the appropriate arguments.
        """
        pass

class TemporalMaskBase(MaskBase, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        parent_inputs = super().INPUT_TYPES()
        return {
            "required": {
                **parent_inputs["required"],
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "end_frame": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "effect_duration": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "temporal_easing": ([ "ease_in_out","linear", "bounce", "elastic", "none"],),
                "palindrome": ("BOOLEAN", {"default": False}),
            }
        }
    
    CATEGORY="RyanOnTheInside/TemporalMasks"

    def __init__(self):
        super().__init__()

    @abstractmethod
    def process_single_mask(self, mask: np.ndarray, strength: float, **kwargs) -> np.ndarray:
        """
        Process a single mask frame. This method must be implemented by child classes.
        frame_index is available in kwargs if needed.
        """
        pass

    def process_mask(self, mask: np.ndarray, strength: float, **kwargs) -> np.ndarray:
        return self.process_single_mask(mask, strength, **kwargs)

    def apply_temporal_mask_operation(self, masks: torch.Tensor, strength: float, start_frame: int, end_frame: int, effect_duration: int, temporal_easing: str, palindrome: bool, **kwargs) -> Tuple[torch.Tensor]:
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        num_frames = masks_np.shape[0]
        
        end_frame = end_frame if end_frame > 0 else num_frames
        effect_duration = min(effect_duration, num_frames) if effect_duration > 0 else (end_frame - start_frame)
        if temporal_easing == "None":
            easing_values = np.ones(effect_duration)
        elif palindrome:
            half_duration = effect_duration // 2
            t = np.linspace(0, 1, half_duration)
            easing_values = apply_easing(t, temporal_easing)
            easing_values = np.concatenate([easing_values, easing_values[::-1]])
        else:
            t = np.linspace(0, 1, effect_duration)
            easing_values = apply_easing(t, temporal_easing)
        
        self.start_progress(num_frames, desc="Applying temporal mask operation")
        
        result = []
        for i in range(num_frames):
            if i < start_frame or i >= end_frame:
                result.append(masks_np[i])
            else:
                frame_in_effect = i - start_frame
                progress = easing_values[frame_in_effect % len(easing_values)]
                temporal_strength = strength * progress
                processed_mask = self.process_single_mask(masks_np[i], temporal_strength, frame_index=i, **kwargs)
                result.append(processed_mask)
            
            self.update_progress()
        
        self.end_progress()
        
        return (torch.from_numpy(np.stack(result)).float(),)

    def main_function(self, masks, strength, invert, subtract_original, grow_with_blur, start_frame, end_frame, effect_duration, temporal_easing, palindrome, **kwargs):
        original_masks = masks
        processed_masks = self.apply_temporal_mask_operation(masks, strength, start_frame, end_frame, effect_duration, temporal_easing, palindrome, **kwargs)
        ret = (self.apply_mask_operation(processed_masks[0], original_masks, strength, invert, subtract_original, grow_with_blur, **kwargs),)
        return ret

    
class OpticalFlowMaskBase(MaskBase, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "images": ("IMAGE",),
                "flow_method": (["Farneback", "LucasKanade", "PyramidalLK"],),
                "flow_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "magnitude_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    CATEGORY="RyanOnTheInside/OpticalFlowMasks"

    def __init__(self):
        super().__init__()

    def process_mask(self, mask: np.ndarray, strength: float, images: np.ndarray, flow_method: str, flow_threshold: float, magnitude_threshold: float, frame_index: int, **kwargs) -> np.ndarray:
        if frame_index == 0 or frame_index >= len(images) - 1:
            return mask

        frame1 = (images[frame_index] * 255).astype(np.uint8)
        frame2 = (images[frame_index + 1] * 255).astype(np.uint8)
        
        flow = calculate_optical_flow(frame1, frame2, flow_method)
        flow_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        flow_magnitude[flow_magnitude < flow_threshold] = 0
        
        flow_magnitude[flow_magnitude < magnitude_threshold * np.max(flow_magnitude)] = 0
        
        flow_magnitude = normalize_array(flow_magnitude)

        return self.apply_flow_mask(mask, flow_magnitude, flow, strength, **kwargs)

    @abstractmethod
    def apply_flow_mask(self, mask: np.ndarray, flow_magnitude: np.ndarray, flow: np.ndarray, strength: float, **kwargs) -> np.ndarray:
        """
        Apply the optical flow-based mask operation. To be implemented by subclasses.
        """
        pass

    def main_function(self, masks, images, strength, flow_method, flow_threshold, magnitude_threshold, **kwargs):
        masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else masks
        images_np = images.cpu().numpy() if isinstance(images, torch.Tensor) else images
        
        num_frames = masks_np.shape[0]
        self.start_progress(num_frames, desc="Applying optical flow mask operation")

        result = []
        for i in range(num_frames):
            processed_mask = self.process_mask(masks_np[i], strength, images_np, flow_method, flow_threshold, magnitude_threshold, frame_index=i, **kwargs)
            result.append(processed_mask)
            self.update_progress()

        self.end_progress()

        processed_masks = np.stack(result)
        return self.apply_mask_operation(processed_masks, masks, strength, **kwargs)

#TODO  check if input mask is blank and just skip, then check children
class FlexMaskBase(MaskBase):
    feature_threshold_default=0.0
    @classmethod
    def INPUT_TYPES(cls):  
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "feature": ("FEATURE",),
                "feature_pipe": ("FEATURE_PIPE",),
                "feature_threshold": ("FLOAT", {"default": cls.feature_threshold_default, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    CATEGORY = "RyanOnTheInside/FlexMasks"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "main_function"

    @abstractmethod
    def process_mask(self, mask: np.ndarray, feature_value: float, strength: float, **kwargs) -> np.ndarray:
        pass

    def apply_mask_operation(self, masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, **kwargs):
        num_frames = feature_pipe.frame_count
        original_masks = masks.clone()

        self.start_progress(num_frames, desc="Applying flex mask operation")

        result = []
        for i in range(num_frames):
            kwargs['frame_index'] = i
            mask = masks[i].numpy()
            feature_value = feature.get_value_at_frame(i)
            
            if feature_value >= feature_threshold:
                processed_mask = self.process_mask(mask, feature_value, strength, **kwargs)
            else:
                if hasattr(self, 'process_mask_below_threshold'):
                    processed_mask = self.process_mask_below_threshold(mask, feature_value, strength, **kwargs)
                else:
                    processed_mask = mask
                

            result.append(processed_mask)
            self.update_progress()

        self.end_progress()

        processed_masks = torch.from_numpy(np.stack(result)).float()
        return super().apply_mask_operation(processed_masks, original_masks, strength, invert, subtract_original, grow_with_blur, **kwargs)

    @abstractmethod
    def main_function(self, masks, feature, feature_pipe, strength, feature_threshold, invert, subtract_original, grow_with_blur, **kwargs):
        pass

