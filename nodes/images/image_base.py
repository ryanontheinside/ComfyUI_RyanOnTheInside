import torch
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from comfy.utils import ProgressBar
from ... import RyanOnTheInside

class FlexImageBase(RyanOnTheInside, ABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feature_param": (cls.get_modifiable_params(), {"default": cls.get_modifiable_params()[0]}),
                "feature_mode": (["relative", "absolute"], {"default": "relative"}),
            },
            "optional": {
                "opt_feature": ("FEATURE",),
                "opt_feature_pipe": ("FEATURE_PIPE",),
            }
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

    def apply_effect(self, images, strength, feature_threshold, feature_param, feature_mode, opt_feature=None, opt_feature_pipe=None, **kwargs):
        if (opt_feature is None) != (opt_feature_pipe is None):
            raise ValueError("Both opt_feature and opt_feature_pipe must be provided together, or neither should be provided.")

        if opt_feature is None and opt_feature_pipe is None:
            # If neither feature nor feature_pipe is provided, process all frames without modulation
            num_frames = images.shape[0]
            images_np = images.cpu().numpy()
            
            self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")
            
            result = []
            for i in range(num_frames):
                processed_image = self.process_image(images_np[i], 0.5, strength, 
                                                     feature_param=feature_param, 
                                                     feature_mode=feature_mode, 
                                                     **kwargs)
                result.append(processed_image)
                self.update_progress()
        else:
            # Original behavior when both feature and feature_pipe are provided
            num_frames = opt_feature_pipe.frame_count
            images_np = images.cpu().numpy()

            self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

            result = []
            for i in range(num_frames):
                image = images_np[i]
                feature_value = opt_feature.get_value_at_frame(i)
                kwargs['frame_index'] = i
                if feature_value >= feature_threshold:
                    processed_image = self.process_image(image, feature_value, strength, 
                                                         feature_param=feature_param, 
                                                         feature_mode=feature_mode, 
                                                         **kwargs)
                else:
                    processed_image = image

                result.append(processed_image)
                self.update_progress()

        self.end_progress()

        # Convert the list of numpy arrays to a single numpy array
        result_np = np.stack(result)
        
        # Convert the numpy array to a PyTorch tensor and ensure it's in BHWC format
        result_tensor = torch.from_numpy(result_np).float()
        
        # If the tensor is not in BHWC format, transpose it
        if result_tensor.shape[1] == 3:  # If it's in BCHW format
            result_tensor = result_tensor.permute(0, 2, 3, 1)

        return (result_tensor,)

    def process_image(self, image: np.ndarray, feature_value: float, strength: float, 
                      feature_param: str, feature_mode: str, **kwargs) -> np.ndarray:
        # Modulate the selected parameter
        for param_name in self.get_modifiable_params():
            if param_name in kwargs:
                if param_name == feature_param:
                    kwargs[param_name] = self.modulate_param(param_name, kwargs[param_name], 
                                                             feature_value, strength, feature_mode)
        
        # Call the child class's implementation
        return self.apply_effect_internal(image, **kwargs)

    @abstractmethod
    def apply_effect_internal(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the effect to the image. To be implemented by child classes."""
        pass
