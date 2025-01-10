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

    CATEGORY = "RyanOnTheInside/FlexImages"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"

    def __init__(self):
        super().__init__()  # Initialize FlexBase
        self.progress_bar = None

    def start_progress(self, total_steps, desc="Processing"):
        self.progress_bar = ProgressBar(total_steps)

    def update_progress(self):
        if self.progress_bar:
            self.progress_bar.update(1)

    def end_progress(self):
        self.progress_bar = None

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
            
            # Set frame index for parameter processing
            kwargs['frame_index'] = i

            # Get feature value (0.5 if no feature provided)
            feature_value = self.get_feature_value(i, opt_feature)
            feature_value = 0.5 if feature_value is None else feature_value

            # Process the image if no feature or feature value meets threshold
            if opt_feature is None or feature_value >= feature_threshold:
                processed_image = self.process_image(
                    image,
                    feature_value,
                    strength,
                    feature_param=feature_param,
                    feature_mode=feature_mode,
                    **kwargs
                )
            else:
                processed_image = image

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

    def process_image(self, image: np.ndarray, feature_value: float, strength: float, 
                     feature_param: str, feature_mode: str, frame_index: int = 0, **kwargs) -> np.ndarray:
        # Process parameters using base class functionality
        processed_kwargs = self.process_parameters(
            frame_index=frame_index,
            feature_value=feature_value,
            strength=strength,
            feature_param=feature_param,
            feature_mode=feature_mode,
            **kwargs
        )
        
        # Call the child class's implementation with processed parameters
        return self.apply_effect_internal(image, **processed_kwargs)
