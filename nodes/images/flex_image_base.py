import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from comfy.utils import ProgressBar
from ... import RyanOnTheInside
from ..flex.flex_base import FlexBase
from ...tooltips import apply_tooltips

@apply_tooltips
class FlexImageBase(RyanOnTheInside, FlexBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_inputs["required"].update({
            "images": ("IMAGE",),
        })
        return base_inputs

    CATEGORY = "RyanOnTheInside/FlexFeatures/Targets/Images"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"

    def __init__(self):
        super().__init__()  # Initialize FlexBase

    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        return []

    def apply_effect(
        self, 
        images, 
        strength, 
        feature_threshold, 
        feature_param, 
        feature_mode, 
        opt_feature=None, 
        **kwargs
    ):
        # Convert images to numpy for processing
        images_np = images.cpu().numpy()

        # Determine frame count from either feature or longest parameter list
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
            # Get the appropriate image frame, handling possible shorter image sequences
            image = images_np[i % images_np.shape[0]]
            
            # Process parameters using base class functionality
            processed_kwargs = self.process_parameters(
                frame_index=i,
                feature_value=self.get_feature_value(i, opt_feature) if opt_feature is not None else None,
                feature_param=feature_param if opt_feature is not None else None,
                feature_mode=feature_mode if opt_feature is not None else None,
                strength=strength,
                feature_threshold=feature_threshold,
                **kwargs
            )
            processed_kwargs["frame_index"] = i
            
            #TODO: Currently dont care about a threshold check here, but may want to add it in the future
            # Apply the effect with processed parameters
            processed_image = self.apply_effect_internal(image, **processed_kwargs)




            result.append(processed_image)
            self.update_progress()

        self.end_progress()

        # Convert the list of numpy arrays to a single numpy array
        result_np = np.stack(result)

        # Convert the numpy array to a PyTorch tensor in BHWC format
        result_tensor = torch.from_numpy(result_np).float()

        return (result_tensor,)

    @abstractmethod
    def apply_effect_internal(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect with processed parameters. To be implemented by child classes."""
        pass
