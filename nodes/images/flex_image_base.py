import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from comfy.utils import ProgressBar
from ... import RyanOnTheInside
from ..flex.flex_base import FlexBase

class FlexImageBase(RyanOnTheInside, FlexBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            **super().INPUT_TYPES(),
            "required": {
                **super().INPUT_TYPES()["required"],
                "images": ("IMAGE",),
            },
            # Optional inputs are inherited from FlexBase
        }

    CATEGORY = "RyanOnTheInside/FlexImage"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"

    def __init__(self):
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

    def modulate_param(self, param_name, param_value, feature_value, strength, mode):
        if mode == "relative":
            return param_value * (1 + (feature_value - 0.5) * strength)
        else:  # absolute
            return param_value * feature_value * strength

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
        # Process all frames without modulation if no feature is provided
        if opt_feature is None:
            num_frames = images.shape[0]
            images_np = images.cpu().numpy()

            self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

            result = []
            for i in range(num_frames):
                processed_image = self.process_image(
                    images_np[i],
                    0.5,
                    strength,
                    feature_param=feature_param,
                    feature_mode=feature_mode,
                    **kwargs
                )
                result.append(processed_image)
                self.update_progress()
        else:
            # Process frames with modulation based on feature values
            images_np = images.cpu().numpy()
            num_frames = opt_feature.frame_count

            self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

            result = []
            for i in range(num_frames):
                image = images_np[i]
                feature_value = opt_feature.get_value_at_frame(i)
                kwargs['frame_index'] = i
                if feature_value >= feature_threshold:
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
        if result_tensor.shape[1] == 3:  # If it's in BCHW format
            result_tensor = result_tensor.permute(0, 2, 3, 1)

        return (result_tensor,)

    def process_image(
        self, 
        image: np.ndarray, 
        feature_value: float, 
        strength: float,
        feature_param: str, 
        feature_mode: str, 
        **kwargs
    ) -> np.ndarray:
        # Modulate the selected parameter
        for param_name in self.get_modifiable_params():
            if param_name in kwargs and param_name == feature_param:
                kwargs[param_name] = self.modulate_param(
                    param_name,
                    kwargs[param_name],
                    feature_value,
                    strength,
                    feature_mode
                )

        # Call the subclass's implementation
        return self.apply_effect_internal(image, **kwargs)

    @abstractmethod
    def apply_effect_internal(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """To be implemented by subclasses."""
        pass
