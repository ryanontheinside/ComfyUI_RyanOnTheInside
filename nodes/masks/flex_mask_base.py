from .mask_base import MaskBase
from abc import ABC, abstractmethod
import numpy as np
import torch

#TODO  SOON TO UPDATE TO SUB FLEXBASE. UNTIL THEN REFER TO IMAGES AUDIO OR VIDEO FOR EXAMPLES ON HOW TO EXTEND
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