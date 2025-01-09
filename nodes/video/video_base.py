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

        result = []
        for i in range(num_frames):
            # Get the appropriate frame, handling possible shorter sequences
            image = images_np[i % images_np.shape[0]]
            
            # Set frame index for parameter processing
            kwargs['frame_index'] = i

            # Get feature value (0.5 if no feature provided)
            feature_value = opt_feature.get_value_at_frame(i) if opt_feature is not None else 0.5

            # Process parameters based on feature value
            processed_kwargs = {}
            for param_name in self.get_modifiable_params():
                if param_name in kwargs:
                    param_value = kwargs[param_name]
                    if isinstance(param_value, (list, tuple, np.ndarray)):
                        try:
                            base_value = float(param_value[i])
                        except (IndexError, TypeError):
                            base_value = float(param_value[0])
                    else:
                        base_value = float(param_value)
                    
                    processed_kwargs[param_name] = self.modulate_param(
                        param_name, base_value, feature_value, strength, feature_mode
                    )

            # Add remaining kwargs
            for key, value in kwargs.items():
                if key not in processed_kwargs and key != 'frame_index':
                    if isinstance(value, (list, tuple, np.ndarray)):
                        try:
                            processed_kwargs[key] = value[i]
                        except (IndexError, TypeError):
                            processed_kwargs[key] = value[0]
                    else:
                        processed_kwargs[key] = value

            # Process the frame
            if opt_feature is None or feature_value >= feature_threshold:
                processed_frame = self.apply_effect_internal(
                    image[np.newaxis, ...],  # Add batch dimension
                    feature_value=feature_value,
                    **processed_kwargs
                )
                result.append(processed_frame[0])  # Remove batch dimension
            else:
                result.append(image)

            self.update_progress()

        self.end_progress()

        # Convert to tensor and ensure BHWC format
        result_tensor = torch.from_numpy(np.stack(result)).float()
        if result_tensor.shape[1] == 3:  # If in BCHW format, convert to BHWC
            result_tensor = result_tensor.permute(0, 2, 3, 1)

        return (result_tensor,)

    @abstractmethod
    def apply_effect_internal(self, video: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect to the entire video. To be implemented by child classes."""
        pass