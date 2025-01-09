import torch
import numpy as np
from abc import ABC, abstractmethod
from ..flex.flex_base import FlexBase
from comfy.utils import ProgressBar
from ...tooltips import apply_tooltips

@apply_tooltips
class FlexVideoBase(FlexBase, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "images": ("IMAGE",),
            }
        }

    CATEGORY = "RyanOnTheInside/FlexBase"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"

    def apply_effect(self, images, strength, feature_mode, feature_threshold, opt_feature=None, **kwargs):
        images_np = images.cpu().numpy()

        # Determine frame count from either feature, images, or longest parameter list
        if opt_feature is not None:
            num_frames = opt_feature.frame_count
        else:
            # Start with number of input frames
            num_frames = images_np.shape[0]
            # Check all parameters for lists/arrays that might be longer
            for value in kwargs.values():
                if isinstance(value, (list, tuple, np.ndarray)):
                    num_frames = max(num_frames, len(value))

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        # First pass: collect all feature values
        feature_values = []
        for i in range(num_frames):
            feature_value = opt_feature.get_value_at_frame(i) if opt_feature is not None else 0.5
            feature_values.append(feature_value)
        feature_values = np.array(feature_values)

        # Process parameters based on feature values #TODO: really should be if  param_name == feature_param
        processed_kwargs = {}
        for param_name in self.get_modifiable_params():
            if param_name in kwargs:
                param_values = []
                for i in range(num_frames):
                    param_value = kwargs[param_name]
                    if isinstance(param_value, (list, tuple, np.ndarray)):
                        try:
                            base_value = float(param_value[i])
                        except (IndexError, TypeError):
                            base_value = float(param_value[0])
                    else:
                        base_value = float(param_value)
                    
                    modulated_value = self.modulate_param(
                        param_name, base_value, feature_values[i], strength, feature_mode
                    )
                    param_values.append(modulated_value)
                processed_kwargs[param_name] = np.array(param_values)

        # Add remaining kwargs
        for key, value in kwargs.items():
            if key not in processed_kwargs and key != 'frame_index':
                if isinstance(value, (list, tuple, np.ndarray)):
                    try:
                        processed_values = [value[i] for i in range(num_frames)]
                    except (IndexError, TypeError):
                        processed_values = [value[0]] * num_frames
                    processed_kwargs[key] = np.array(processed_values)
                else:
                    processed_kwargs[key] = value

        # Process the entire video at once
        if opt_feature is None or np.all(feature_values >= feature_threshold):
            processed_video = self.apply_effect_internal(
                images_np,
                feature_values=feature_values,
                opt_feature=opt_feature,
                **processed_kwargs
            )
        else:
            # Only apply effect to frames where feature value is above threshold
            mask = feature_values >= feature_threshold
            processed_video = images_np.copy()
            if np.any(mask):
                processed_frames = self.apply_effect_internal(
                    images_np[mask],
                    feature_values=feature_values[mask],
                    opt_feature=opt_feature,
                    **{k: v[mask] if isinstance(v, np.ndarray) else v for k, v in processed_kwargs.items()}
                )
                processed_video[mask] = processed_frames

        self.end_progress()

        # Convert to tensor and ensure BHWC format
        result_tensor = torch.from_numpy(processed_video).float()
        if result_tensor.shape[1] == 3:  # If in BCHW format, convert to BHWC
            result_tensor = result_tensor.permute(0, 2, 3, 1)

        return (result_tensor,)

    @abstractmethod
    def apply_effect_internal(self, video: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect to the entire video. To be implemented by child classes."""
        pass